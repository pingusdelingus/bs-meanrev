from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.utils
import json
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import time
from functools import lru_cache

app = Flask(__name__)

@lru_cache(maxsize=1)
def get_spx_options():
    return 0
    """
    Get current SPX price and 1-month ATM options with caching
    """
    try:
        # Add delay to respect rate limits
        #time.sleep(1)
        
        #spx = yf.Ticker("^GSPC")
        #current_price = spx.info.get('regularMarketPrice', 0)
        current_price = 0
        if current_price == 0:
            return {
                'current_price': 'N/A',
                'atm_call': 'N/A',
                'atm_put': 'N/A',
                'error': 'Unable to fetch current price'
            }
        
        # Get options chain with error handling
        try:
            #options = spx.option_chain()
            #calls = options.calls
            #puts = options.puts
            calls = []
            puts = []
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return {
                'current_price': round(current_price, 2),
                'atm_call': 'N/A',
                'atm_put': 'N/A',
                'error': 'Unable to fetch options data'
            }
        
        # Find ATM options (closest to current price)
        calls['strike_diff'] = abs(calls['strike'] - current_price)
        puts['strike_diff'] = abs(puts['strike'] - current_price)
        
        try:
            atm_call = calls.loc[calls['strike_diff'].idxmin(), 'lastPrice']
            atm_put = puts.loc[puts['strike_diff'].idxmin(), 'lastPrice']
        except Exception as e:
            print(f"Error finding ATM options: {e}")
            return {
                'current_price': round(current_price, 2),
                'atm_call': 'N/A',
                'atm_put': 'N/A',
                'error': 'Unable to find ATM options'
            }
        
        return {
            'current_price': round(current_price, 2),
            'atm_call': round(atm_call, 2),
            'atm_put': round(atm_put, 2),
            'error': None
        }
    except Exception as e:
        print(f"Error fetching SPX data: {e}")
        return {
            'current_price': 'N/A',
            'atm_call': 'N/A',
            'atm_put': 'N/A',
            'error': str(e)
        }

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate Black-Scholes option price
    S: Current stock price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free interest rate
    sigma: Volatility
    option_type: 'call' or 'put'
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return price

def simulate_gbm(S, T, r, sigma, num_steps=1000):
    """
    Simulate Geometric Brownian Motion
    """
    dt = T/num_steps
    t = np.linspace(0, T, num_steps)
    
    # Generate random walk
    W = np.random.standard_normal(num_steps)
    W = np.cumsum(W)*np.sqrt(dt)
    
    # Calculate stock price path
    S_t = S * np.exp((r - sigma**2/2)*t + sigma*W)
    
    return t, S_t

def calculate_risk_neutral_price(S, K, T, r, sigma, num_paths=1000):
    """
    Calculate option price using risk-neutral pricing theorem
    """
    dt = T/100  # Time steps
    paths = np.zeros((num_paths, 101))  # 101 points including initial price
    paths[:, 0] = S
    
    # Simulate multiple paths under risk-neutral measure
    for i in range(100):
        paths[:, i+1] = paths[:, i] * np.exp((r - sigma**2/2)*dt + sigma*np.sqrt(dt)*np.random.standard_normal(num_paths))
    
    # Calculate payoffs
    payoffs = np.maximum(paths[:, -1] - K, 0)
    option_price = np.exp(-r*T) * np.mean(payoffs)
    
    return paths, payoffs, option_price

def heston_model_simulation(S0, K, T, r, v0, kappa, theta, sigma, rho, num_paths=1000, num_steps=100):
    """
    Simulate Heston stochastic volatility model
    S0: Initial stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    v0: Initial variance
    kappa: Mean reversion speed
    theta: Long-run variance
    sigma: Volatility of variance
    rho: Correlation between stock and variance processes
    """
    dt = T/num_steps
    paths = np.zeros((num_paths, num_steps + 1))
    variance_paths = np.zeros((num_paths, num_steps + 1))
    
    # Initialize paths
    paths[:, 0] = S0
    variance_paths[:, 0] = v0
    
    # Generate correlated random numbers
    z1 = np.random.standard_normal((num_paths, num_steps))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal((num_paths, num_steps))
    
    # Simulate paths
    for t in range(num_steps):
        # Ensure variance stays positive
        variance_paths[:, t] = np.maximum(variance_paths[:, t], 0)
        
        # Update stock price
        paths[:, t+1] = paths[:, t] * np.exp((r - 0.5*variance_paths[:, t])*dt + 
                                            np.sqrt(variance_paths[:, t]*dt) * z1[:, t])
        
        # Update variance
        variance_paths[:, t+1] = variance_paths[:, t] + kappa*(theta - variance_paths[:, t])*dt + \
                                sigma*np.sqrt(variance_paths[:, t]*dt) * z2[:, t]
    
    # Calculate option payoffs
    payoffs = np.maximum(paths[:, -1] - K, 0)
    option_price = np.exp(-r*T) * np.mean(payoffs)
    
    return paths, variance_paths, option_price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spx_info')
def spx_info():
    return jsonify(get_spx_options())

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    
    # Extract parameters
    S = float(data['stock_price'])
    K = float(data['strike_price'])
    T = float(data['time_to_maturity'])
    r = float(data['risk_free_rate'])
    sigma = float(data['volatility'])
    
    # Calculate option price
    option_price = black_scholes(S, K, T, r, sigma)
    
    # Simulate stock price path
    t, S_t = simulate_gbm(S, T, r, sigma)
    
    # Create stock price plot with dark theme
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=S_t,
        mode='lines',
        name='Stock Price',
        line=dict(color='#89b4fa')
    ))
    fig.add_trace(go.Scatter(
        x=[0, T],
        y=[K, K],
        mode='lines',
        name='Strike Price',
        line=dict(color='#f5c2e7', dash='dash')
    ))
    fig.update_layout(
        title='Stock Price Simulation (Geometric Brownian Motion)',
        xaxis_title='Time (years)',
        yaxis_title='Stock Price',
        showlegend=True,
        paper_bgcolor='#313244',
        plot_bgcolor='#1e1e2e',
        font=dict(color='#cdd6f4'),
        xaxis=dict(
            gridcolor='#89b4fa',
            color='#cdd6f4'
        ),
        yaxis=dict(
            gridcolor='#89b4fa',
            color='#cdd6f4'
        )
    )
    
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'option_price': round(option_price, 4),
        'plot': plot_json
    })

@app.route('/risk_neutral', methods=['POST'])
def risk_neutral():
    data = request.get_json()
    
    # Extract parameters
    S = float(data['stock_price'])
    K = float(data['strike_price'])
    T = float(data['time_to_maturity'])
    r = float(data['risk_free_rate'])
    sigma = float(data['volatility'])
    
    # Calculate risk-neutral price and paths
    paths, payoffs, option_price = calculate_risk_neutral_price(S, K, T, r, sigma)
    
    # Create static visualization
    fig = go.Figure()
    
    # Add sample paths
    for i in range(10):  # Show 10 sample paths
        fig.add_trace(go.Scatter(
            y=paths[i],
            mode='lines',
            name=f'Path {i+1}',
            line=dict(color='#89b4fa', width=1)
        ))
    
    # Add strike price line
    fig.add_trace(go.Scatter(
        y=[K] * 101,
        mode='lines',
        name='Strike Price',
        line=dict(color='#f5c2e7', dash='dash')
    ))
    
    fig.update_layout(
        title='Risk-Neutral Pricing Simulation',
        xaxis_title='Time Steps',
        yaxis_title='Stock Price',
        showlegend=True,
        paper_bgcolor='#313244',
        plot_bgcolor='#1e1e2e',
        font=dict(color='#cdd6f4'),
        xaxis=dict(
            gridcolor='#89b4fa',
            color='#cdd6f4'
        ),
        yaxis=dict(
            gridcolor='#89b4fa',
            color='#cdd6f4'
        )
    )
    
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'option_price': round(option_price, 4),
        'plot': plot_json
    })

@app.route('/heston', methods=['POST'])
def heston():
    data = request.get_json()
    
    # Extract basic parameters
    S = float(data['stock_price'])
    K = float(data['strike_price'])
    T = float(data['time_to_maturity'])
    r = float(data['risk_free_rate'])
    v0 = float(data['volatility'])**2  # Convert volatility to variance
    
    # Heston specific parameters (could be made configurable through UI)
    kappa = 2.0    # Mean reversion speed
    theta = v0     # Long-run variance (set to initial variance for simplicity)
    sigma = 0.3    # Volatility of variance
    rho = -0.7    # Correlation (typically negative for equity)
    
    # Calculate Heston model paths and price
    paths, variance_paths, option_price = heston_model_simulation(S, K, T, r, v0, kappa, theta, sigma, rho)
    
    # Create visualization
    fig = go.Figure()
    
    # Add sample paths
    for i in range(10):  # Show 10 sample paths
        fig.add_trace(go.Scatter(
            y=paths[i],
            mode='lines',
            name=f'Path {i+1}',
            line=dict(color='#89b4fa', width=1)
        ))
    
    # Add strike price line
    fig.add_trace(go.Scatter(
        y=[K] * (len(paths[0])),
        mode='lines',
        name='Strike Price',
        line=dict(color='#f5c2e7', dash='dash')
    ))
    
    # Add mean path
    mean_path = np.mean(paths, axis=0)
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode='lines',
        name='Mean Path',
        line=dict(color='#a6e3a1', width=2)
    ))
    
    fig.update_layout(
        title='Heston Model Simulation',
        xaxis_title='Time Steps',
        yaxis_title='Stock Price',
        showlegend=True,
        paper_bgcolor='#313244',
        plot_bgcolor='#1e1e2e',
        font=dict(color='#cdd6f4'),
        xaxis=dict(
            gridcolor='#89b4fa',
            color='#cdd6f4'
        ),
        yaxis=dict(
            gridcolor='#89b4fa',
            color='#cdd6f4'
        )
    )
    
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'option_price': round(option_price, 4),
        'plot': plot_json
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 

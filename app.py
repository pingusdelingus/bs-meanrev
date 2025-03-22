from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import json

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

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
    
    # Create stock price plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=S_t,
        mode='lines',
        name='Stock Price',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=[0, T],
        y=[K, K],
        mode='lines',
        name='Strike Price',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title='Stock Price Simulation (Geometric Brownian Motion)',
        xaxis_title='Time (years)',
        yaxis_title='Stock Price',
        showlegend=True
    )
    
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return jsonify({
        'option_price': round(option_price, 4),
        'plot': plot_json
    })

if __name__ == '__main__':
    app.run(debug=True) 
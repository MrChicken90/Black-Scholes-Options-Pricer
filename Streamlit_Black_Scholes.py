import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Black-Scholes Option Pricer & P&L",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Black-Scholes Calculation Functions ---
def black_scholes_price(S, K, T, r, sigma, option_type):
    """Calculates the Black-Scholes option price without greeks."""
    # Ensure sigma is not zero to avoid division by zero
    if sigma <= 0:
        sigma = 1e-10 # Use a very small number instead of zero
    if T <= 0:
        T = 1e-10

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def black_scholes_with_greeks(S, K, T, r, sigma, option_type):
    """Calculates price and greeks."""
    # Ensure sigma is not zero to avoid division by zero
    if sigma <= 0:
        sigma = 1e-10 # Use a very small number instead of zero
    if T <= 0:
        T = 1e-10

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    elif option_type == 'put':
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    return {
        'Price': price, 'Delta': delta, 'Gamma': gamma, 
        'Theta (per day)': theta, 'Vega (per 1%)': vega, 'Rho (per 1%)': rho
    }

# --- Sidebar for User Inputs ---
st.sidebar.header("Option Parameters")

S = st.sidebar.number_input("Current Asset Price ($)", min_value=1.0, value=100.0, step=0.5)
K = st.sidebar.number_input("Strike Price ($)", min_value=1.0, value=105.0, step=0.5)
T_days = st.sidebar.number_input("Time to Maturity (Days)", min_value=1, max_value=730, value=30, step=1)
T = T_days / 365.0
r_percent = st.sidebar.number_input("Annual Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1)
r = r_percent / 100
sigma_percent = st.sidebar.number_input("Annual Volatility (σ) (%)", min_value=1.0, max_value=100.0, value=20.0, step=0.5)
sigma = sigma_percent / 100

st.sidebar.header("Trade Parameters")
option_type_selection = st.sidebar.radio("Option Type for P&L Heatmap", ('call', 'put'))
purchase_price = st.sidebar.number_input(f"Purchase Price of {option_type_selection.title()} Option ($)", min_value=0.0, value=1.50, step=0.01)

# --- Main Panel Display ---
st.title("Black-Scholes Option Pricing and P&L Analysis")
st.markdown("---")

# --- Value & Greeks Display ---
col1, col2 = st.columns(2)

# Column 1: Call and Put Values
with col1:
    st.header("Option Values")
    call_price = black_scholes_price(S, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S, K, T, r, sigma, 'put')
    
    # Custom HTML for styled metric-like boxes
    st.markdown(f"""
    <div style="background-color: rgba(40, 167, 69, 0.2); border: 1px solid rgba(40, 167, 69, 0.5); border-radius: 7px; padding: 15px; margin-bottom: 10px;">
        <div style="font-size: 0.9rem; color: #808495;">Call Option Value</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${call_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: rgba(220, 53, 69, 0.2); border: 1px solid rgba(220, 53, 69, 0.5); border-radius: 7px; padding: 15px;">
        <div style="font-size: 0.9rem; color: #808495;">Put Option Value</div>
        <div style="font-size: 1.5rem; font-weight: bold;">${put_price:.2f}</div>
    </div>
    """, unsafe_allow_html=True)


# Column 2: Greeks
with col2:
    st.header("Option Greeks")
    greeks = black_scholes_with_greeks(S, K, T, r, sigma, option_type_selection)
    
    # Display greeks in two sub-columns for better layout
    sub_col1, sub_col2 = st.columns(2)
    sub_col1.metric(label="Delta", value=f"{greeks['Delta']:.3f}")
    sub_col1.metric(label="Gamma", value=f"{greeks['Gamma']:.3f}")
    sub_col1.metric(label="Vega (per 1%)", value=f"{greeks['Vega (per 1%)']:.3f}")
    sub_col2.metric(label="Theta (per day)", value=f"{greeks['Theta (per day)']:.3f}")
    sub_col2.metric(label="Rho (per 1%)", value=f"{greeks['Rho (per 1%)']:.3f}")
    
st.markdown("---")

# --- P&L Heatmap Generation ---
st.header(f"Profit & Loss (P&L) Heatmap for a {option_type_selection.title()} Option")

# Define the grid for the heatmap based on current inputs
spot_price_range = np.linspace(S * 0.8, S * 1.2, 10)
volatility_range = np.linspace(sigma * 0.8, sigma * 1.2, 10)

# Handle case where volatility is zero
if volatility_range.min() <= 0:
    volatility_range = np.linspace(0.01, sigma * 1.2, 10)


# Calculate P&L matrix
pnl_matrix = np.zeros((len(volatility_range), len(spot_price_range)))
for i, vol in enumerate(volatility_range):
    for j, spot in enumerate(spot_price_range):
        price_at_scenario = black_scholes_price(spot, K, T, r, vol, option_type_selection)
        pnl = price_at_scenario - purchase_price
        pnl_matrix[i, j] = pnl

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    pnl_matrix,
    xticklabels=[f"${s:.2f}" for s in spot_price_range],
    yticklabels=[f"{v:.2%}" for v in volatility_range],
    annot=True,
    fmt=".2f",
    cmap='RdYlGn',
    ax=ax,
    annot_kws={"size": 10}
)
ax.invert_yaxis()
ax.set_title(f'P&L for {option_type_selection.title()} (K=${K:.2f}, Paid=${purchase_price:.2f})', fontsize=16)
ax.set_xlabel('Spot Price', fontsize=12)
ax.set_ylabel('Volatility', fontsize=12)

st.pyplot(fig)

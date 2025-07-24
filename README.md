# Black-Scholes Option Pricer & P&L Heatmap

This is an interactive web application built with Streamlit that calculates European option prices, Greeks, and visualizes potential Profit & Loss (P&L) scenarios using the Black-Scholes model.

## ✨ Features

- **Interactive Inputs**: Adjust all key option parameters (stock price, strike, time, volatility, risk-free rate) via a user-friendly sidebar.
- **Greeks Analysis**: View the primary option Greeks (Delta, Gamma, Vega, Theta, Rho) to understand the option's risk profile.
- **P&L Heatmap**: Visualize your potential profit or loss for a trade across a grid of future stock prices and volatility levels.

## 🚀 Running Locally

To run this application on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install the dependencies:**
    Make sure you have a `requirements.txt` file with the necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run Streamlit_Black_Scholes.py
    ```

The application will open in your default web browser.

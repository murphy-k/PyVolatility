import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def simulate_GARCH(n, omega, alpha, beta = 0):
    np.random.seed(1)
    white_noise = np.random.normal(size=n)
    resid = np.zeros_like(white_noise)
    variance = np.zeros_like(white_noise)

    for t in range(1,n):
        variance[t] = omega + alpha * resid[t-1]**2 + beta * variance[t-1]
        resid[t] = np.sqrt(variance[t]) * white_noise[t] 

    return resid, variance


df = pd.read_csv(r'C:\Users\JCA49\PyVolatility\SPX Index.csv',index_col='Date',parse_dates=True)
df['Return'] = (df['SPX Index'].pct_change()*100)
df['Realized Variance'] = df['Return']**2

st.title('GARCH Model Simulator')

st.sidebar.header("Model Parameters")
n = st.sidebar.slider('Number of Observations', min_value=30,max_value=1000,value=252,step=1)
omega = st.sidebar.slider('Omega', min_value=0.0,max_value=1.0,value=0.1,step=0.1)
alpha = st.sidebar.slider('Alpha',min_value=0.0,max_value=1.0,value=0.7, step=0.1)
beta = st.sidebar.slider('Beta',min_value=0.0,max_value=1.0,value=0.1, step=0.1)

# Simulate ARCH series
arch_resid, arch_variance = simulate_GARCH(n=n,omega=omega, alpha=alpha)

garch_resid, garch_variance = simulate_GARCH(n=n, omega=omega, alpha=alpha, beta=beta)

fig, ax = plt.subplots(figsize=(1.618*8,8))
ax.plot(arch_variance, color='black',label='ARCH Variance')
ax.plot(garch_variance, color='red',label='GARCH Variance')
ax.legend()
ax.set_title("ARCH vs GARCH Variance")
ax.set_xlabel("Time")
ax.set_ylabel('Variance')

st.pyplot(fig)

st.write("""
         This plot shows the simulated variance for both ARCH(1) and GARCH(1,1) models. 
         Adjust the parameters using the sliders in the sidebar to see how they affect variance over time.
         
         Parameter Constraints:
         * All parameters are non-negative, so the variance cannot be negative. 
         omega, alpha, and beta > 0
         
         * Model estimations are "mean-reverting" to the long-run variance.
         alpha + beta < 1
         ---
         """
         )

st.write("""
         Actual Realized Variance of the SP500"""
         )

fig2, ax2 = plt.subplots(figsize=(1.618*8,8))
ax2.plot(df['Realized Variance'])
ax2.set_title("Realized Variance of the SP500")
ax2.set_xlabel('Date')
ax2.set_ylabel("Realized Variance")
st.pyplot(fig2)
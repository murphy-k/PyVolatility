import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_GARCH(n, omega, alpha, beta = 0):
    np.random.seed(4)
    # Initialize the parameters
    white_noise = np.random.normal(size = n)
    resid = np.zeros_like(white_noise)
    variance = np.zeros_like(white_noise)
    
    for t in range(1, n):
        # Simulate the variance (sigma squared)
        variance[t] = omega + alpha * resid[t-1]**2 + beta * variance[t-1]
        # Simulate the residuals
        resid[t] = np.sqrt(variance[t]) * white_noise[t]    
    
    return resid, variance

st.title('GARCH Model Simulator')

st.sidebar.header('Model Parameters')
n = st.sidebar.slider('Number of observations', min_value=100, max_value=1000, value=252, step=1)
omega = st.sidebar.slider('Omega', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
alpha = st.sidebar.slider('Alpha', min_value=0.0, max_value=1.0, value=0.7, step=0.1)
beta = st.sidebar.slider('Beta', min_value=0.0, max_value=1.0, value=0.1, step=0.1)

# Simulate a ARCH(1) series
arch_resid, arch_variance = simulate_GARCH(n=n, omega=omega, alpha=alpha)

# Simulate a GARCH(1,1) series
garch_resid, garch_variance = simulate_GARCH(n=n, omega=omega, alpha=alpha, beta=beta)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(arch_variance, color='red', label='ARCH Variance')
ax.plot(garch_variance, color='orange', label='GARCH Variance')
ax.legend()
ax.set_title('ARCH vs GARCH Variance')
ax.set_xlabel('Time')
ax.set_ylabel('Variance')

# Display the plot in Streamlit
st.pyplot(fig)

st.write("""
This plot shows the simulated variance for both ARCH(1) and GARCH(1,1) models.
- The red line represents the ARCH(1) variance.
- The orange line represents the GARCH(1,1) variance.

You can adjust the parameters using the sliders in the sidebar to see how they affect the variance over time.
""")
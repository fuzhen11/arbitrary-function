import numpy as np
import matplotlib.pyplot as plt

# Define the Mexican Hat (Ricker) wavelet function
def mexican_hat(t):
    return (1 - t**2) * np.exp(-t**2 / 2)

# Generate a range of values for t
t = np.linspace(-5, 5, 500)  # Interval [-5, 5] with 500 points

# Calculate the wavelet values
wavelet_values = mexican_hat(t)

# Plot the Mexican Hat wavelet
plt.figure(figsize=(8, 4))
plt.plot(t, wavelet_values, label="Mexican Hat Wavelet", color="purple")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Horizontal reference line
plt.title("Mexican Hat (Ricker) Wavelet")
plt.xlabel("t")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

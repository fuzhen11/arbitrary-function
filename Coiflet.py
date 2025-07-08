import pywt
import numpy as np
import matplotlib.pyplot as plt

# Define the Coiflet wavelet
wavelet_name = "coif3"  # Coiflet wavelet with 3 vanishing moments
wavelet = pywt.Wavelet(wavelet_name)

# Get the scaling and wavelet functions
phi, psi, x = wavelet.wavefun(level=10)  # Higher level for better resolution

# Plot the scaling and wavelet functions
plt.figure(figsize=(12, 6))

# Plot the scaling function
plt.subplot(1, 2, 1)
plt.plot(x, phi, label="Scaling function (φ)", color="blue")
plt.title(f"Scaling Function of {wavelet_name}")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# Plot the wavelet function
plt.subplot(1, 2, 2)
plt.plot(x, psi, label="Wavelet function (ψ)", color="red")
plt.title(f"Wavelet Function of {wavelet_name}")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

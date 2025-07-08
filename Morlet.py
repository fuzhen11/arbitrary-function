import numpy as np
import matplotlib.pyplot as plt

# Define the Morlet wavelet function
def morlet(t, w=5.0):
    """
    Morlet wavelet function.
    Parameters:
        t: Time variable (array).
        w: Central frequency of the wavelet (default is 5.0).
    Returns:
        Real and imaginary parts of the Morlet wavelet.
    """
    gaussian = np.exp(-t**2 / 2)
    wave = np.exp(1j * w * t) * gaussian
    return wave.real, wave.imag

# Generate a range of time values
t = np.linspace(-5, 5, 500)  # Interval [-5, 5] with 500 points

# Compute the real and imaginary parts of the Morlet wavelet
real_part, imag_part = morlet(t)

# Plot the Morlet wavelet
plt.figure(figsize=(10, 5))

# Real part
plt.subplot(2, 1, 1)
plt.plot(t, real_part, label="Real Part", color="blue")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Horizontal reference line
plt.title("Morlet Wavelet - Real Part")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

# Imaginary part
plt.subplot(2, 1, 2)
plt.plot(t, imag_part, label="Imaginary Part", color="orange")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Horizontal reference line
plt.title("Morlet Wavelet - Imaginary Part")
plt.xlabel("Time (t)")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

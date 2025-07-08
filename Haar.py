import numpy as np
import matplotlib.pyplot as plt

# Create the Haar wavelet function
def haar_wavelet(t, scale=0.5):
    result = np.piecewise(t,
                          [(0 <= t) & (t < scale), (scale <= t) & (t < 2*scale), (t >= 2*scale) | (t<0)],
                          [lambda t: -1,
                           lambda t: 1,
                           lambda t: 0])
    return result


# Define the time variable
t = np.linspace(-1, 2, 1000)

x_array = np.array(t)

# Generate the Haar wavelet
wavelet = haar_wavelet(x_array, scale=0.5)

# Plot the Haar wavelet
plt.plot(t, wavelet)
plt.title('Haar Wavelet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

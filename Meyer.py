import numpy as np
import matplotlib.pyplot as plt


def meyer_wavelet(t):

    def phi_hat(w):
        # frequency function
        abs_w = np.abs(w)
        if abs_w < 2 * np.pi / 3:
            return 1
        elif 2 * np.pi / 3 <= abs_w <= 4 * np.pi / 3:
            return np.cos(np.pi / 2 * np.log2(3 * abs_w / (2 * np.pi)))
        else:
            return 0

    # frequency range
    freq_range = np.linspace(-2 * np.pi, 2 * np.pi, len(t))
    phi_values = np.array([phi_hat(w) for w in freq_range])

    # inverse fourier transform
    phi_time = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(phi_values))).real
    return phi_time


# time point
time = np.linspace(-3, 3, 50)
meyer_wavelet_values = meyer_wavelet(time)

# plot Meyer
plt.plot(time, meyer_wavelet_values)
plt.title("Meyer Wavelet (Time Domain)")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

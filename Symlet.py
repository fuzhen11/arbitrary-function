import pywt
import matplotlib.pyplot as plt

# Symlets 'sym4'
wavelet_name = 'sym4'

wavelet = pywt.Wavelet(wavelet_name)

phi, psi, x = wavelet.wavefun(level=5)

# plot
plt.figure(figsize=(12, 6))

#  phi
plt.subplot(2, 1, 1)
plt.plot(x, phi, label="Scaling Function (phi)", color='blue')
plt.title(f"Scaling Function (phi) of {wavelet_name}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

#  psi
plt.subplot(2, 1, 2)
plt.plot(x, psi, label="Wavelet Function (psi)", color='orange')
plt.title(f"Wavelet Function (psi) of {wavelet_name}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


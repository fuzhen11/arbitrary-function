import numpy as np
import matplotlib.pyplot as plt

def battle_lemarie_wavelet(x):

    return (np.exp(-x**2) * np.cos(10 * x))


x = np.linspace(-10, 10, 1000)

y = battle_lemarie_wavelet(x)

# plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Battle-Lemarié Wavelet", color='b')
plt.axhline(0, color='k', linewidth=0.5, linestyle="--")
plt.axvline(0, color='k', linewidth=0.5, linestyle="--")
plt.title("Battle-Lemarié Wavelet")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.grid(True)
plt.show()

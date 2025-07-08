import numpy as np
import matplotlib.pyplot as plt
import pywt
import random

def wavelet_function(x, wavelet_type, A=1, B=1, C=1):
    wavelets = {
        1: lambda x: A * np.sin(B * x + C),  # Sin
        2: lambda x: A * np.cos(B * x + C),  # Cos
        3: lambda x: A * np.exp(-B * x ** 2),  # Gaussian
        4: lambda x: A * np.sign(np.sin(B * x)),  # Square wave
        5: lambda x: A * np.tanh(B * x + C),  # Hyperbolic tangent
        6: lambda x: A * np.exp(-B * np.abs(x)) * np.cos(C * x),  # Enhanced damped sine
        7: lambda x: A * (x ** 2) * np.exp(-B * x ** 2),  # Enhanced quadratic gaussian
        8: lambda x: A * (np.sinc(B * (x - C))),  # Sinc
        9: lambda x: A * np.heaviside(x - C, 0.5),  # Step
        10: lambda x: A * (np.exp(-x**2) * np.cos(B * x)),  # Battle-Lemarié
        11: lambda x: A * (1 - x**2) * np.exp(-x**2 / B),  # Mexican Hat
        12: lambda x: Haar(x,A,B,C), #Haar
        13: lambda x: Meyer(x,A,B,C), #Meyer,C is not used
        14: lambda x: Morlet(x,A,B,C), #Morlet
        15: lambda x: Daubechies(x,A,B,C), #Daubechies,B,C are not used
        16: lambda x: Coiflet(x,A,B,C), #Coiflet,B,C are not used
        17: lambda x: Symlet(x,A,B,C), #Symlet,B,C are not used
        18: lambda x: Beylkin(x,A,B,C), #Beylkin,B,C are not used
        19: lambda x: Random(x,A,B,C)  #Random, B,C are not used
    }
    return wavelets.get(wavelet_type, lambda x: np.zeros_like(x))(x)

def Haar(x,A,B,C):
    result = np.piecewise(x,
                          [(0 <= x) & (x < B), (B <= x) & (x < 2 * B), (x >= 2 * B) | (x < 0)],
                          [lambda x: -1*A,
                           lambda x: 1*A,
                           lambda x: 0])
    return result

def Meyer(x,A,B,C):

    def phi_hat(w,A):
        # frequency function
        abs_w = np.abs(w)
        if abs_w < 2 * np.pi / 3:
            return 1*A
        elif 2 * np.pi / 3 <= abs_w <= 4 * np.pi / 3:
            return A*np.cos(np.pi / 2 * np.log2(3 * abs_w / (2 * np.pi)))
        else:
            return 0

    # frequency range
    freq_range = np.linspace(-B * np.pi, B * np.pi, len(x))
    phi_values = np.array([phi_hat(w,A) for w in freq_range])

    # inverse fourier transform
    phi_time = np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(phi_values))).real
    return phi_time

def Morlet(x,A,B,C):
    gaussian = np.exp(-(x-C)**2 / 2)
    wave = A*np.exp(1j * B * (x-C)) * gaussian
    #Here,we use the real part, sure, the imag part also can be selected.
    return wave.real

def Daubechies(x,A,B,C):
    # Define the wavelet
    wavelet_name = "db2"  # Daubechies wavelet with 4 vanishing moments
    wavelet = pywt.Wavelet(wavelet_name)

    # Get the scaling and wavelet functions
    phi, psi, xx = wavelet.wavefun(level=5)  # Higher level for better resolution
    xxx = min(xx) + (x-min(x))*(max(xx)-min(xx))/(max(x)-min(x))# linear mapping
    xxx_phi = A*np.interp(xxx, xx, phi)
    xxx_psi = A*np.interp(xxx, xx, psi)
    return xxx_psi

def Coiflet(x,A,B,C):
    # Define the Coiflet wavelet
    wavelet_name = "coif3"  # Coiflet wavelet with 3 vanishing moments
    wavelet = pywt.Wavelet(wavelet_name)

    # Get the scaling and wavelet functions
    phi, psi, xx = wavelet.wavefun(level=5)  # Higher level for better resolution
    xxx = min(xx) + (x - min(x)) * (max(xx) - min(xx)) / (max(x) - min(x))  # linear mapping
    xxx_phi = A * np.interp(xxx, xx, phi)
    xxx_psi = A * np.interp(xxx, xx, psi)
    return xxx_psi

def Symlet(x,A,B,C):
    # Define the Symlet wavelet
    wavelet_name = "sym4"  # Symlet wavelet w
    wavelet = pywt.Wavelet(wavelet_name)

    # Get the scaling and wavelet functions
    phi, psi, xx = wavelet.wavefun(level=5)  # Higher level for better resolution
    xxx = min(xx) + (x - min(x)) * (max(xx) - min(xx)) / (max(x) - min(x))  # linear mapping
    xxx_phi = A * np.interp(xxx, xx, phi)
    xxx_psi = A * np.interp(xxx, xx, psi)
    return xxx_psi

def Beylkin(x,A,B,C):
    # Define the Beylkin wavelet
    wavelet_name = "db10"  # Beylkin wavelet
    wavelet = pywt.Wavelet(wavelet_name)

    # Get the scaling and wavelet functions
    phi, psi, xx = wavelet.wavefun(level=5)  # Higher level for better resolution
    xxx = min(xx) + (x - min(x)) * (max(xx) - min(xx)) / (max(x) - min(x))  # linear mapping
    xxx_phi = A * np.interp(xxx, xx, phi)
    xxx_psi = A * np.interp(xxx, xx, psi)
    return xxx_psi

def Random(x,A,B,C):
    result=	[A*random.random() for _ in x]
    return result

def calculate_wavelet(wavelet_type, a, b, num=101, A=1, B=1, C=1):
    x = np.linspace(a, b, num)
    y = wavelet_function(x, wavelet_type, A, B, C)
    return x,y

def plot_wavelet(x,y):
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Waveform data in one period')
    plt.grid()
    plt.show()
def output_wavelet_data(x,y):
    #matrix = np.column_stack((x, y))
    # save to txt file at the same path with code
    output_file = "output_matrix.txt"
    np.savetxt(output_file, y, fmt='%.8f', delimiter='\t')

if __name__ == "__main__":
    wavelet_type = int(input("select function (1-19): "))
    a = float(input("input a (Min): "))
    b = float(input("input b (Max): "))
    num = int(input("input num (Number of data): "))
    A = float(input("input A (Amplitude): "))
    B = float(input("input B (Parameter 1): "))
    C = float(input("input C (Parameter 1): "))
    x,y = calculate_wavelet(wavelet_type, a, b, num, A, B, C)
    plot_wavelet(x,y)
    output_wavelet_data(x, y)

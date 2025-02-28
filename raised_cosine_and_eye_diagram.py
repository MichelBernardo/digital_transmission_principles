import numpy as np
import matplotlib.pyplot as plt

def raised_cosine_pulse(t, T, r):
    """
    Gera um pulso de cosseno levantado.
    t: tempo
    T: período de símbolo
    r: fator de roll-off
    """
    sinc_part = np.sinc(t / T)  # sin(np.pi*t/T)/(np.pi*t/T)
    cos_part = np.cos(np.pi * r * t / T)
    denom = 1 - (2 * r * t / T) ** 2
    
    # Avoid divide by zero 
    pulse = np.where(denom != 0, sinc_part * cos_part / denom, np.pi / 4 * np.sinc(1 / (2 * r)))
    return pulse

def generate_signal(N, T, r, modulation="BPSK", M=2, Fs=1000):
    """
    Gera um sinal composto por N pulsos de cosseno levantado espaçados por T segundos.
    N: número de pulsos
    T: período de símbolo
    r: fator de roll-off
    modulation: Tipo de modulação ("BPSK" ou "PAM")
    M: Número de símbolos na modulação PAM (deve ser potência de 2)
    Fs: frequência de amostragem
    """
    t_total = (N + 150) * T  # Duração total do sinal
    t = np.linspace(-t_total / 2, t_total / 2, int(Fs * t_total))
    signal = np.zeros_like(t)
    pulses = []
    
    # Gerar sequência de símbolos aleatória
    bits = np.random.randint(0, M, N)
    
    # Mapear bits para símbolos conforme a modulação escolhida
    if modulation == "BPSK":
        symbols = 2 * bits - 1  # -1 e +1
    elif modulation == "PAM":
        symbols = np.array([2 * m - 1 - M for m in (bits + 1)])
    else:
        raise ValueError("Modulação inválida. Escolha 'BPSK' ou 'PAM'")
    
    for n in range(N):
        pulse = symbols[n] * raised_cosine_pulse(t - n * T, T, r)  # Ajusta pulso para refletir símbolo
        pulses.append(pulse)
        signal += pulse
    
    return t, signal, pulses, bits, modulation

def plot_eye_diagram(signal, T, Fs):
    """
    Plota o diagrama do olho para o sinal recebido.
    signal: sinal modulado
    T: período do símbolo
    Fs: frequência de amostragem
    """
    samples_per_symbol = int(Fs * T)
    num_segments = len(signal) // samples_per_symbol
    
    plt.figure(figsize=(8, 6))
    for i in range(num_segments - 1):
        segment = signal[i * samples_per_symbol:(i + 2) * samples_per_symbol]
        time_axis = np.linspace(0, 2 * T, len(segment))
        plt.plot(time_axis, segment, color='b', alpha=0.3)
    
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.title("Diagrama do Olho")
    plt.grid()
    plt.show()

# Parâmetros
N = 50      # Número de pulsos
T = 1      # Período de símbolo
r = 1    # Fator de roll-off
Fs = 1000  # Frequência de amostragem
modulation = "PAM"  # Escolha "BPSK" ou "PAM"
M = 4      # Número de símbolos na modulação PAM

t, signal, pulses, bits, modulation = generate_signal(N, T, r, modulation, M, Fs)

# Exibir sequência de bits gerada
print(f"Modulação: {modulation} com {M} níveis")
print("Sequência de símbolos transmitida:", bits)

bit_sequence = ''.join(format(b, f'0{int(np.log2(M))}b') for b in bits)
print("Sequência de bits real:", bit_sequence)

# Plotar os pulsos individuais
plt.figure(figsize=(10, 6))
for i, pulse in enumerate(pulses):
    plt.plot(t, pulse)
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title(f"Pulsos Individuais de Cosseno Levantado ({modulation})")
plt.legend()
plt.grid()
plt.show()

# Plotar o sinal total
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label=f"Sinal com {N} pulsos modulados ({modulation})")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title(f"Sinal de Pulsos de Cosseno Levantado ({modulation})")
plt.legend()
plt.grid()
plt.show()

# Gerar e plotar o diagrama do olho
plot_eye_diagram(signal, T, Fs)
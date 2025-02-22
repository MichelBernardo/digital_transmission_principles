import numpy as np
import matplotlib.pyplot as plt

def raised_cosine_pulse(t, T, r):
    """
    Gera um pulso de cosseno levantado.
    t: tempo
    T: período de símbolo
    r: fator de roll-off
    """
    sinc_part = np.sinc(t / T)
    cos_part = np.cos(np.pi * r * t / T)
    denom = 1 - (2 * r * t / T) ** 2
    
    # Evitar divisão por zero
    pulse = np.where(denom != 0, sinc_part * cos_part / denom, np.pi / 4 * np.sinc(1 / (2 * r)))
    return pulse

def generate_signal(N, T, r, Fs=1000):
    """
    Gera um sinal composto por N pulsos de cosseno levantado espaçados por T segundos.
    N: número de pulsos
    T: período de símbolo
    r: fator de roll-off
    Fs: frequência de amostragem
    """
    t_total = (N + 100) * T  # Duração total do sinal
    t = np.linspace(-t_total / 2, t_total / 2, int(Fs * t_total))
    signal = np.zeros_like(t)
    pulses = []
    
    # Gerar sequência de bits aleatória (0s e 1s)
    bits = np.random.randint(0, 2, N)
    symbols = 2 * bits - 1  # Mapear para -1 e +1 (BPSK)
    
    for n in range(N):
        pulse = symbols[n] * raised_cosine_pulse(t - n * T, T, r)  # Ajusta pulso para refletir bit
        pulses.append(pulse)
        signal += pulse
    
    return t, signal, pulses, bits

# Parâmetros
N = 25      # Número de pulsos
T = 1      # Período de símbolo
r = 0.5    # Fator de roll-off
Fs = 1000  # Frequência de amostragem

t, signal, pulses, bits = generate_signal(N, T, r, Fs)

# Exibir sequência de bits gerada
print("Sequência de bits transmitida:", bits)

# Plotar os pulsos individuais
plt.figure(figsize=(10, 6))
for i, pulse in enumerate(pulses):
    plt.plot(t, pulse, label=f"Pulso {i+1} (Bit {bits[i]})")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("Pulsos Individuais de Cosseno Levantado (Modulados pelos Bits)")
plt.legend()
plt.grid()
plt.show()

# Plotar o sinal total
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label=f"Sinal com {N} pulsos modulados")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.title("Sinal de Pulsos de Cosseno Levantado (Modulado pelos Bits)")
plt.legend()
plt.grid()
plt.show()

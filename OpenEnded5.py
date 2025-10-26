import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt
from numpy.fft import rfft, rfftfreq

file_path = "ppg_100hz_1024samples.csv"
fs = 15.0

try:
    df = pd.read_csv(file_path)
except:
    df = pd.read_csv(file_path, delim_whitespace=True, header=None)


if df.shape[1] == 1:
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
    except:
        df = pd.read_csv(file_path, delim_whitespace=True, header=None)

print("Detected columns:", df.columns.tolist())
print("Data shape:", df.shape)

if df.shape[1] == 1:
    print("Only one column detected — assuming it's a single PPG channel.")
    df.columns = ['PPG']
    red = df['PPG'].astype(float).values
    ir = df['PPG'].astype(float).values
else:
    red = df.iloc[:, 0].astype(float).values
    ir = df.iloc[:, 1].astype(float).values

N = len(red)
t = np.arange(N) / fs

plt.figure(figsize=(10,4))
plt.plot(t, red, label='RED / PPG', color='red', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Raw PPG Signal")
plt.legend()
plt.grid(True)
plt.show()

freqs = rfftfreq(N, 1/fs)
red_fft = np.abs(rfft(red - np.mean(red)))
ir_fft = np.abs(rfft(ir - np.mean(ir)))

plt.figure(figsize=(10,4))
plt.plot(freqs, red_fft, label='RED Spectrum', color='red')
plt.plot(freqs, ir_fft, label='IR Spectrum', color='purple')
plt.xlim(0, fs/2)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title("Magnitude of DFT vs Frequency")
plt.legend()
plt.grid(True)
plt.show()

mask = (freqs >= 0.5) & (freqs <= 4.0)
peak_freq = freqs[mask][np.argmax(red_fft[mask])]
heart_rate_bpm = peak_freq * 60

print(f"Estimated Heart Rate: {heart_rate_bpm:.2f} bpm")
if np.allclose(red, ir):
    print("Only one channel available — cannot compute SpO₂.")
else:
    AC_red = np.std(detrend(red))
    AC_ir  = np.std(detrend(ir))
    DC_red = np.mean(red)
    DC_ir  = np.mean(ir)
    R = (AC_red/DC_red) / (AC_ir/DC_ir)
    SpO2 = np.clip(110 - 25*R, 0, 100)
    print(f"Estimated SpO₂ = {SpO2:.2f}%")


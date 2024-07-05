import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft

def calculate_parameters(file_path, column_index=1, start_from=0, target_frequency=80, target_bandwidth=1):
    fs = 250  # Sampling frequency

    df = pd.read_csv(file_path, delimiter='\t')
    if df.shape[1] <= column_index:
        raise ValueError(f"The file {file_path} does not have column index {column_index}.")

    x = df.iloc[start_from:, column_index].dropna()

    xdft_full = fft(x)
    xdft_full = xdft_full[:len(x)//2 + 1]
    psdx_full = (1 / (fs * len(x))) * np.abs(xdft_full)**2
    psdx_full[1:-1] = 2 * psdx_full[1:-1]

    freq_full = np.linspace(0, fs/2, len(psdx_full))
    target_freq_indices_full = (freq_full >= (target_frequency - target_bandwidth / 2)) & (freq_full <= (target_frequency + target_bandwidth / 2))
    target_peak_power_full = np.max(psdx_full[target_freq_indices_full])
    target_peak_index_full = np.argmax(psdx_full[target_freq_indices_full])
    peak_freq_full = freq_full[target_freq_indices_full][target_peak_index_full]

    noise_indices_full = (freq_full >= (target_frequency - 5)) & (freq_full <= (target_frequency + 5)) & \
                         ~((freq_full >= peak_freq_full - 0.5) & (freq_full <= peak_freq_full + 0.5))
    noise_powers_full = psdx_full[noise_indices_full]
    noise_power_avg_full = np.mean(noise_powers_full)
    SNR_full = 10 * np.log10(target_peak_power_full / noise_power_avg_full)

    target_50Hz_indices_full = (freq_full >= 49) & (freq_full <= 51)
    target_50Hz_power_full = np.max(psdx_full[target_50Hz_indices_full])
    target_50Hz_index_full = np.argmax(psdx_full[target_50Hz_indices_full])
    peak_50Hz_full = freq_full[target_50Hz_indices_full][target_50Hz_index_full]

    eighty_to_fifty_ratio = 10 * np.log10(target_peak_power_full) - 10 * np.log10(target_50Hz_power_full)

    return (SNR_full, peak_freq_full, 10 * np.log10(target_peak_power_full),
            10 * np.log10(noise_power_avg_full), 10 * np.log10(target_50Hz_power_full), eighty_to_fifty_ratio,
            freq_full, psdx_full, peak_50Hz_full)



def plot_results(freq_full, psdx_full, peak_freq_full, target_peak_power_full, peak_50Hz_full, target_50Hz_power_full, SNR_full, noise_power_avg_full, eighty_to_fifty_ratio, file_name, plot_title="Periodogram Using FFT", y_range=None):
    plt.figure()
    plt.plot(freq_full, 10 * np.log10(psdx_full))
    if y_range:
        plt.ylim(y_range)
    plt.grid(True)
    plt.title(plot_title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    
    plt.scatter(peak_freq_full, 10 * np.log10(target_peak_power_full), color='red', marker='*', s=100)
    plt.text(peak_freq_full + 1, 10 * np.log10(target_peak_power_full), f'Peak: ({peak_freq_full:.2f} Hz, {10 * np.log10(target_peak_power_full):.2f} dB/Hz)', horizontalalignment='left')
    
    plt.scatter(peak_50Hz_full, 10 * np.log10(target_50Hz_power_full), color='red', marker='*', s=100)
    plt.text(peak_50Hz_full - 1, 10 * np.log10(target_50Hz_power_full), f'Peak: ({peak_50Hz_full:.2f} Hz, {10 * np.log10(target_50Hz_power_full):.2f} dB/Hz)', horizontalalignment='right')

    textstr = '\n'.join((
        f'SNR: {SNR_full:.2f} dB',
        f'Average Noise Power: {10 * np.log10(noise_power_avg_full):.2f} dB/Hz',
        f'80Hz to 50Hz Ratio: {eighty_to_fifty_ratio:.2f} dB',
    ))

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.figtext(0.5, -0.05, file_name, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()
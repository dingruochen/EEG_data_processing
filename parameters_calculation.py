import pandas as pd
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import os

def plot_transient(file_path, column_index=1, start_from=0, ylim=None, title="EEG Signal time domain"):
    fs = 250  # Sampling frequency
    df = pd.read_csv(file_path, delimiter='\t')
    if df.shape[1] <= column_index:
        raise ValueError(f"The file {file_path} does not have column index {column_index}.")

    x = df.iloc[start_from:, column_index].dropna()

    samples = range(start_from + 1, start_from + len(x) + 1)

    plt.figure()
    plt.plot(samples, x)
    plt.xlabel('Samples')
    plt.ylabel('EEG Signal')

    if ylim:
        plt.ylim(ylim)

    plt.title(title)
    plt.grid(True)
    plt.show()


def calculate_parameters(file_path, column_index=1, start_from=0, target_frequency=12.5, target_bandwidth=1, noise_range=(5, 20),delimiter='\t'):
    fs = 250  # Sampling frequency

    df = pd.read_csv(file_path, delimiter=delimiter)
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

    noise_indices_full = (freq_full >= noise_range[0]) & (freq_full <= noise_range[1]) & \
                         ~((freq_full >= peak_freq_full - 0.5) & (freq_full <= peak_freq_full + 0.5))
    noise_powers_full = psdx_full[noise_indices_full]
    noise_power_avg_full = np.mean(noise_powers_full)
    SNR_full = 10 * np.log10(target_peak_power_full / noise_power_avg_full) if noise_power_avg_full > 0 else np.nan

    target_50Hz_indices_full = (freq_full >= 49) & (freq_full <= 51) & \
                               ~((freq_full >= peak_freq_full - 0.5) & (freq_full <= peak_freq_full + 0.5))
    target_50Hz_power_full = np.max(psdx_full[target_50Hz_indices_full])
    target_50Hz_index_full = np.argmax(psdx_full[target_50Hz_indices_full])
    peak_50Hz_full = freq_full[target_50Hz_indices_full][target_50Hz_index_full]

    Target_to_50Hz_ratio = 10 * np.log10(target_peak_power_full) - 10 * np.log10(target_50Hz_power_full) if target_50Hz_power_full > 0 else np.nan

    return (SNR_full, peak_freq_full, 10 * np.log10(target_peak_power_full) if target_peak_power_full > 0 else np.nan,
            10 * np.log10(noise_power_avg_full) if noise_power_avg_full > 0 else np.nan, 
            10 * np.log10(target_50Hz_power_full) if target_50Hz_power_full > 0 else np.nan, 
            Target_to_50Hz_ratio, freq_full, psdx_full, peak_50Hz_full)

def plot_results(ax, freq_full, psdx_full, peak_freq_full, target_peak_power_full, peak_50Hz_full, target_50Hz_power_full, SNR_full, noise_power_avg_full, Target_to_50Hz_ratio, file_name, plot_title="Periodogram Using FFT", y_range=None):
    ax.plot(freq_full, 10 * np.log10(psdx_full))
    if y_range:
        ax.set_ylim(y_range)
    ax.grid(True)
    ax.set_title(plot_title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power/Frequency (dB/Hz)")
    
    if np.isfinite(target_peak_power_full):
        ax.scatter(peak_freq_full, 10 * np.log10(target_peak_power_full), color='red', marker='*', s=100)
        ax.text(peak_freq_full, 10 * np.log10(target_peak_power_full), f'Peak: ({peak_freq_full:.2f} Hz, {10 * np.log10(target_peak_power_full):.2f} dB/Hz)')
    else:
        print(f"Invalid peak power value for {file_name}: {target_peak_power_full}")
    
    if np.isfinite(target_50Hz_power_full):
        ax.scatter(peak_50Hz_full, 10 * np.log10(target_50Hz_power_full), color='red', marker='*', s=100)
        ax.text(peak_50Hz_full, 10 * np.log10(target_50Hz_power_full), f'Peak: ({peak_50Hz_full:.2f} Hz, {10 * np.log10(target_50Hz_power_full):.2f} dB/Hz)')
    else:
        print(f"Invalid 50Hz power value for {file_name}: {target_50Hz_power_full}")

    textstr = '\n'.join((
        f'SNR: {SNR_full:.2f} dB' if np.isfinite(SNR_full) else 'SNR: N/A',
        f'Average Noise Powers: {10 * np.log10(noise_power_avg_full):.2f} dB/Hz' if np.isfinite(noise_power_avg_full) else 'Average Noise Powers: N/A',
        f'Target to 50Hz ratio: {Target_to_50Hz_ratio:.2f} dB' if np.isfinite(Target_to_50Hz_ratio) else 'Target to 50Hz ratio: N/A'
    ))

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)

def process_files(base_dir, file_names, column_indices=[1], start_from=2, target_frequency=12.5, target_bandwidth=1, noise_range=(5, 20), plot_title="Periodogram Using FFT", y_range=(-140, -20),delimiter='\t'):
    results = []
    file_paths = [os.path.join(base_dir, file_name) for file_name in file_names]

    # Collect results
    for file_path in file_paths:
        for column_index in column_indices:
            result = calculate_parameters(file_path, column_index, start_from, target_frequency, target_bandwidth, noise_range,delimiter)
            if result:
                snr, peak_freq, peak_power, noise_power_avg, peak_50Hz_power,target_to_50Hz_ratio, freq_full, psdx_full, peak_50Hz_full = result
                results.append([f"{os.path.basename(file_path)} (CH{column_index})", snr, peak_freq, peak_power, noise_power_avg, peak_50Hz_power, target_to_50Hz_ratio])

    headers = ["Parameters", "SNR (dB)", "Peak Frequency (Hz)", "Peak Power (dB/Hz)", "Average Noise Power (dB/Hz)", "Peak 50Hz Power (dB/Hz)", "Target f to 50Hz Ratio (dB)"]

    # Create and print DataFrame
    df = pd.DataFrame(results, columns=headers)
    df_transposed = df.T
    df_transposed.columns = df_transposed.iloc[0]
    df_transposed = df_transposed[1:]

    print(df_transposed.to_string())

    # Plot results
    for file_path in file_paths:
        fig, axes = plt.subplots(1, len(column_indices), figsize=(6 * len(column_indices), 5))
        if len(column_indices) == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        for ax, column_index in zip(axes, column_indices):
            result = calculate_parameters(file_path, column_index, start_from, target_frequency, target_bandwidth, noise_range,delimiter)
            if result:
                snr, peak_freq, peak_power, noise_power_avg, peak_50Hz_power, target_to_50Hz_ratio, freq_full, psdx_full, peak_50Hz_full = result
                
                plot_results(ax, freq_full, psdx_full, peak_freq, 10 ** (peak_power / 10), peak_50Hz_full, 10 ** (peak_50Hz_power / 10), snr, 10 ** (noise_power_avg / 10), target_to_50Hz_ratio, file_name=plot_title, plot_title=plot_title, y_range=y_range)
                
                ax.text(0.5, -0.2, f"{os.path.basename(file_path)} (CH{column_index})", transform=ax.transAxes, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.show()

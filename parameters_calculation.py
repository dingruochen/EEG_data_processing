import pandas as pd
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import os

def plot_transient(file_path, column_index=1, start_from=0, ylim=None):
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

    plt.title('EEG Signal time domain for small needle electrodes')
    plt.grid(True)
    plt.show()


def calculate_parameters(file_path, column_index=1, start_from=0, target_frequency=12.5, target_bandwidth=1):
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

    noise_indices_full = (freq_full >= 5) & (freq_full <= 20) & \
                         ~((freq_full >= peak_freq_full - 0.5) & (freq_full <= peak_freq_full + 0.5))
    noise_powers_full = psdx_full[noise_indices_full]
    noise_power_avg_full = np.mean(noise_powers_full)
    SNR_full = 10 * np.log10(target_peak_power_full / noise_power_avg_full)

    target_50Hz_indices_full = (freq_full >= 49) & (freq_full <= 51) & \
                               ~((freq_full >= peak_freq_full - 0.5) & (freq_full <= peak_freq_full + 0.5))
    target_50Hz_power_full = np.max(psdx_full[target_50Hz_indices_full])
    target_50Hz_index_full = np.argmax(psdx_full[target_50Hz_indices_full])
    peak_50Hz_full = freq_full[target_50Hz_indices_full][target_50Hz_index_full]

    SSVEP_to_50Hz_ratio = 10 * np.log10(target_peak_power_full) - 10 * np.log10(target_50Hz_power_full)

    return (SNR_full, peak_freq_full, 10 * np.log10(target_peak_power_full),
            10 * np.log10(noise_power_avg_full), 10 * np.log10(target_50Hz_power_full), SSVEP_to_50Hz_ratio,
            freq_full, psdx_full, peak_50Hz_full)


def plot_results(freq_full, psdx_full, peak_freq_full, target_peak_power_full, peak_50Hz_full, target_50Hz_power_full, SNR_full, noise_power_avg_full, SSVEP_to_50Hz_ratio, file_name, plot_title="Periodogram Using FFT", y_range=None):
    plt.figure()
    plt.plot(freq_full, 10 * np.log10(psdx_full))
    if y_range:
        plt.ylim(y_range)
    plt.grid(True)
    plt.title(plot_title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.scatter(peak_freq_full, 10 * np.log10(target_peak_power_full), color='red', marker='*', s=100)
    plt.text(peak_freq_full, 10 * np.log10(target_peak_power_full), f'Peak: ({peak_freq_full:.2f} Hz, {10 * np.log10(target_peak_power_full):.2f} dB/Hz)')
    plt.scatter(peak_50Hz_full, 10 * np.log10(target_50Hz_power_full), color='red', marker='*', s=100)
    plt.text(peak_50Hz_full, 10 * np.log10(target_50Hz_power_full), f'Peak: ({peak_50Hz_full:.2f} Hz, {10 * np.log10(target_50Hz_power_full):.2f} dB/Hz)')

    textstr = '\n'.join((
        f'SNR: {SNR_full:.2f} dB',
        f'Average Noise Powers: {10 * np.log10(noise_power_avg_full):.2f} dB',
        f'SSVEP to 50Hz ratio: {SSVEP_to_50Hz_ratio:.2f} dB'
    ))

    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
                   verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.figtext(0.5, -0.05, file_name, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


def process_files(base_dir, file_names, column_index=1, start_from=2, target_frequency=12.5, target_bandwidth=1, plot_title="Periodogram Using FFT", y_range=(-140, -20)):
    # array to save the results
    results = []
    file_paths = [os.path.join(base_dir, file_name) for file_name in file_names]

    for file_path in file_paths:
        result = calculate_parameters(file_path, column_index, start_from, target_frequency, target_bandwidth)
        if result:
            snr, peak_freq, peak_power, noise_power_avg, peak_50Hz_power, SSVEP_to_50Hz_Ratio, freq_full, psdx_full, peak_50Hz_full = result
            
            results.append([os.path.basename(file_path), round(snr, 2), round(peak_freq, 2), round(peak_power, 2), round(noise_power_avg, 2), round(peak_50Hz_power, 2), round(SSVEP_to_50Hz_Ratio, 2)])

    headers = ["File Name", "SNR (dB)", "Peak Frequency (Hz)", "Peak Power (dB/Hz)", "Average Noise Power (dB/Hz)", "Peak 50Hz Power (dB/Hz)", "80Hz to 50Hz Ratio (dB)"]
    column_width = 30

    # print column names
    print("".ljust(column_width), end="")
    for result in results:
        print(result[0].ljust(column_width), end="")
    print("\n" + "-" * (column_width * (len(results) + 1)))

    # print results
    for i in range(1, len(headers)):
        print(headers[i].ljust(column_width), end="")
        for result in results:
            print(str(result[i]).ljust(column_width), end="")
        print()

    # plot
    for file_path in file_paths:
        result = calculate_parameters(file_path, column_index, start_from, target_frequency, target_bandwidth)
        if result:
            snr, peak_freq, peak_power, noise_power_avg, peak_50Hz_power, SSVEP_to_50Hz_Ratio, freq_full, psdx_full, peak_50Hz_full = result
            
            plot_results(freq_full, psdx_full, peak_freq, 10 ** (peak_power / 10), peak_50Hz_full, 10 ** (peak_50Hz_power / 10), snr, 10 ** (noise_power_avg / 10), SSVEP_to_50Hz_Ratio, file_name=os.path.basename(file_path), plot_title=plot_title, y_range=y_range)
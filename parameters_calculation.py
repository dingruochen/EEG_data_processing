# data_analysis.py
import pandas as pd
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt



def process_with_plot(file_path, column_index=1, start_from=0, plot_title="Periodogram Using FFT", y_range=None):
    fs = 250  # Sampling frequency

    try:
        # 使用制表符作为分隔符读取CSV文件
        df = pd.read_csv(file_path, delimiter='\t')
        if df.shape[1] <= column_index:
            raise ValueError(f"The file {file_path} does not have column index {column_index}.")

        # 从指定列获取数据，从指定点开始读取
        x = df.iloc[start_from:, column_index].dropna()  # 确保数据在正确的列，并从start_from开始

        xdft_full = fft(x)
        xdft_full = xdft_full[:len(x)//2 + 1]
        psdx_full = (1 / (fs * len(x))) * np.abs(xdft_full)**2
        psdx_full[1:-1] = 2 * psdx_full[1:-1]

        freq_full = np.linspace(0, fs/2, len(psdx_full))
        target_freq_indices_full = (freq_full >= 12.2) & (freq_full <= 12.8)
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

        plt.figure()
        plt.plot(freq_full, 10 * np.log10(psdx_full))
        if y_range:
            plt.ylim(y_range)  # 设置y轴范围
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

        plt.show()

        print(f"Processed {file_path}: "
              f"SNR={SNR_full:.2f} dB, "
              f"Peak Frequency={peak_freq_full:.2f} Hz, "
              f"Peak Power={10 * np.log10(target_peak_power_full):.2f} dB, "
              f"Noise Power Avg={10 * np.log10(noise_power_avg_full):.2f} dB, "
              f"50Hz Peak Power={10 * np.log10(target_50Hz_power_full):.2f} dB, "
              f"SSVEP to 50Hz ratio={SSVEP_to_50Hz_ratio:.2f} dB")

        return (SNR_full, peak_freq_full, 10 * np.log10(target_peak_power_full),
                10 * np.log10(noise_power_avg_full), 10 * np.log10(target_50Hz_power_full), SSVEP_to_50Hz_ratio)

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None


def processing_no_plot(file_path, column_index=1, start_from=0):
    fs = 250  # Sampling frequency

    try:
        # 使用制表符作为分隔符读取CSV文件
        df = pd.read_csv(file_path, delimiter='\t')
        if df.shape[1] <= column_index:
            raise ValueError(f"The file {file_path} does not have column index {column_index}.")

        # 从指定列获取数据，从指定点开始读取
        x = df.iloc[start_from:, column_index].dropna()  # 确保数据在正确的列，并从start_from开始

        xdft_full = fft(x)
        xdft_full = xdft_full[:len(x)//2 + 1]
        psdx_full = (1 / (fs * len(x))) * np.abs(xdft_full)**2
        psdx_full[1:-1] = 2 * psdx_full[1:-1]

        freq_full = np.linspace(0, fs/2, len(psdx_full))
        target_freq_indices_full = (freq_full >= 9.8) & (freq_full <= 10.3)
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
        #peak_50Hz_full = freq_full[target_50Hz_indices_full][target_50Hz_index_full]

        SSVEP_to_50Hz_radio = 10 * np.log10(target_peak_power_full)-10 * np.log10(target_50Hz_power_full)

        print(f"Processed {file_path}: "
              f"SNR={SNR_full:.2f} dB, "
              f"Peak Frequency={peak_freq_full:.2f} Hz, "
              f"Peak Power={10 * np.log10(target_peak_power_full):.2f} dB, "
              f"Noise Power Avg={10 * np.log10(noise_power_avg_full):.2f} dB, "
              f"50Hz Peak Power={10 * np.log10(target_50Hz_power_full):.2f} dB,"
              f"SSVEP_to_50Hz_radio={SSVEP_to_50Hz_radio:.2f} dB")

        return (SNR_full, peak_freq_full, 10 * np.log10(target_peak_power_full),
                10 * np.log10(noise_power_avg_full), 10 * np.log10(target_50Hz_power_full),SSVEP_to_50Hz_radio)

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None
    


def plot_transient(file_path, column_index=1, start_from=0, ylim=None):
    fs = 250  # Sampling frequency

    try:
        # 使用制表符作为分隔符读取CSV文件
        df = pd.read_csv(file_path, delimiter='\t')
        if df.shape[1] <= column_index:
            raise ValueError(f"The file {file_path} does not have column index {column_index}.")

        # 从指定列获取数据，从start_from行开始
        x = df.iloc[start_from:, column_index].dropna()  # 确保数据在正确的列，并从start_from开始读取

        # 生成样本索引，考虑到起始点不为0
        samples = range(start_from + 1, start_from + len(x) + 1)

        # 绘制图形
        plt.figure()
        plt.plot(samples, x)
        plt.xlabel('Samples')
        plt.ylabel('EEG Signal')

        # 设置 y 轴范围
        if ylim:
            plt.ylim(ylim)

        plt.title('EEG Signal time domain for small needle electrodes')
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.fft import rfft, rfftfreq

def process_with_sliding_window_plot(file_path, column_index=1, start_from=0, fs=250, target_freq=80, window_size=10000, step_size=250, window_type='hann'):
    df = pd.read_csv(file_path, delimiter='\t')
    x = df.iloc[start_from:, column_index].dropna()
    x = x.to_numpy()  # 确保x是numpy数组

    # 凯塞窗
    window = np.kaiser(window_size, beta=1)  

    psd_avg = np.zeros(window_size // 2 + 1)
    count = 0

    # 计算滑动窗口FFT
    for start in range(0, len(x) - window_size + 1, step_size):
        window_data = x[start:start + window_size] * window
        xdft = rfft(window_data)
        psdx = (1 / (fs * window_size)) * np.abs(xdft)**2
        psdx[1:-1] *= 2  # 除直流和尼奎斯特分量外，其余成分乘以2
        psd_avg += psdx
        count += 1

    psd_avg /= count
    freq = rfftfreq(window_size, 1 / fs)
    psd_avg_db = 10 * np.log10(psd_avg + 1e-12)  # 防止对0取对数

    # 计算80Hz和50Hz的峰值以及SNR
    target_freq_indices = (freq >= target_freq - 0.5) & (freq <= target_freq + 0.5)
    target_peak_power = np.max(psd_avg[target_freq_indices])
    target_peak_freq = freq[target_freq_indices][np.argmax(psd_avg[target_freq_indices])]

    noise_indices = (freq >= 75) & (freq <= 85) & ~((freq >= target_peak_freq - 0.5) & (freq <= target_peak_freq + 0.5))
    noise_power_avg = np.mean(psd_avg[noise_indices])

    SNR = 10 * np.log10(target_peak_power / noise_power_avg)

    fifty_hz_indices = (freq >= 49) & (freq <= 51)
    fifty_hz_peak_power = np.max(psd_avg[fifty_hz_indices])

    # 计算50Hz峰值功率与80Hz峰值功率的比值
    fifty_to_eighty_ratio = 10 * np.log10(fifty_hz_peak_power) - 10 * np.log10(target_peak_power)

    # 可视化频谱
    plt.figure()
    plt.plot(freq, psd_avg_db)
    plt.scatter(target_peak_freq, 10 * np.log10(target_peak_power), color='red', label='80 Hz Peak')
    plt.scatter(50, 10 * np.log10(fifty_hz_peak_power), color='blue', label='50 Hz Peak')
    plt.title("Power Spectrum Using Sliding Window with Kaiser Window")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return SNR, target_peak_freq, 10 * np.log10(target_peak_power), 10 * np.log10(noise_power_avg), 10 * np.log10(fifty_hz_peak_power), fifty_to_eighty_ratio


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
        target_freq_indices_full = (freq_full >= 79.5) & (freq_full <= 80.5)
        target_peak_power_full = np.max(psdx_full[target_freq_indices_full])
        target_peak_index_full = np.argmax(psdx_full[target_freq_indices_full])
        peak_freq_full = freq_full[target_freq_indices_full][target_peak_index_full]

        noise_indices_full = (freq_full >= 75) & (freq_full <= 85) & \
                             ~((freq_full >= peak_freq_full - 0.5) & (freq_full <= peak_freq_full + 0.5))
        noise_powers_full = psdx_full[noise_indices_full]
        noise_power_avg_full = np.mean(noise_powers_full)
        SNR_full = 10 * np.log10(target_peak_power_full / noise_power_avg_full)

        target_50Hz_indices_full = (freq_full >= 49) & (freq_full <= 51)
        target_50Hz_power_full = np.max(psdx_full[target_50Hz_indices_full])
        target_50Hz_index_full = np.argmax(psdx_full[target_50Hz_indices_full])
        peak_50Hz_full = freq_full[target_50Hz_indices_full][target_50Hz_index_full]

        eighty_to_fifty_ratio = 10 * np.log10(target_peak_power_full) - 10 * np.log10(target_50Hz_power_full)

        plt.figure()
        plt.plot(freq_full, 10 * np.log10(psdx_full))
        if y_range:
            plt.ylim(y_range)  # 设置y轴范围
        plt.grid(True)
        plt.title(plot_title)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power/Frequency (dB/Hz)")
        
        # 80Hz标注点放在右边
        plt.scatter(peak_freq_full, 10 * np.log10(target_peak_power_full), color='red', marker='*', s=100)
        plt.text(peak_freq_full + 1, 10 * np.log10(target_peak_power_full), f'Peak: ({peak_freq_full:.2f} Hz, {10 * np.log10(target_peak_power_full):.2f} dB/Hz)', horizontalalignment='left')
        
        # 50Hz标注点向上移动更多
        plt.scatter(peak_50Hz_full, 10 * np.log10(target_50Hz_power_full), color='red', marker='*', s=100)
        plt.text(peak_50Hz_full - 1, 10 * np.log10(target_50Hz_power_full) , f'Peak: ({peak_50Hz_full:.2f} Hz, {10 * np.log10(target_50Hz_power_full):.2f} dB/Hz)', horizontalalignment='right')

        textstr = '\n'.join((
            f'SNR: {SNR_full:.2f} dB',
            f'Average Noise Power: {10 * np.log10(noise_power_avg_full):.2f} dB/Hz',
            f'80Hz to 50Hz Ratio: {eighty_to_fifty_ratio:.2f} dB',
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
              f"80Hz to 50Hz Ratio={eighty_to_fifty_ratio:.2f} dB")

        return (SNR_full, peak_freq_full, 10 * np.log10(target_peak_power_full),
                10 * np.log10(noise_power_avg_full), 10 * np.log10(target_50Hz_power_full), eighty_to_fifty_ratio)

    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")
        return None
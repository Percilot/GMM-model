import wave
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from python_speech_features import mfcc, delta

def wavReader(wavFile, wavArrayOutputPath, comparisonChartOutputPath, mfccOutputPath):
    fs = 48000

    # 打开 WAV 文件
    wav_file = wave.open(wavFile, 'r')

    # 读取 WAV 文件的参数
    num_channels = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()

    # 读取 WAV 文件的数据
    wav_data = wav_file.readframes(num_frames)

    # 关闭 WAV 文件
    wav_file.close()

    # 将 WAV 数据转换为 numpy 数组
    wav_np = np.frombuffer(wav_data, dtype=np.int16)

    # 根据声道数和采样宽度重新整理数据
    if num_channels > 1:
        wav_np = wav_np.reshape(-1, num_channels)
        wav_np = np.mean(wav_np, axis=1)  # 取平均值以得到单声道数据
    
    if num_frames < fs:
        padding_length = fs - num_frames
        wav_np = np.pad(wav_np, (0, padding_length), mode='constant')
    elif num_frames > fs:
        wav_np = wav_np[:fs]
    else:
        wav_np = wav_np

    np.savetxt(wavArrayOutputPath, wav_np)

    # 设计 IIR 滤波器
    order = 1  # 滤波器阶数
    cutoff_freq = 5000.0  # 截止频率（以 Hz 为单位）
    b, a = signal.butter(order, cutoff_freq, fs=frame_rate, btype='low', analog=False)

    # 应用滤波器
    filtered_wav = signal.lfilter(b, a, wav_np)

    # 绘制滤波前后的波形
    time = np.arange(0, len(wav_np)) / frame_rate
    fig, axes = plt.subplots(2)
    axes[0].plot(time, wav_np)
    axes[0].set_title('Original Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(time, filtered_wav)
    axes[1].set_title('Filtered Waveform')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')

    plt.tight_layout()
    plt.savefig(comparisonChartOutputPath)
    plt.close()

    mfcc0 = mfcc(signal=filtered_wav, samplerate=fs, nfft=2048)
    mfcc1 = delta(mfcc0, 1)
    mfcc2 = delta(mfcc0, 2)
    mfcc_all = np.hstack((mfcc0, mfcc1, mfcc2))
    np.savetxt(mfccOutputPath, mfcc_all, delimiter=',')

if __name__ == "__main__":
    for root, dirs, files in os.walk(sys.argv[1]):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            wavReader(file_path, sys.argv[2] + '/' + file_name + '.txt', sys.argv[3] + '/' + file_name + '.png', sys.argv[4] + '/' + file_name + '.txt')


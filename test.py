import os
import torch
import librosa
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from src import (
    resample_oct,
    _symmetric_toeplitz_numpy,
)


def test_resample():
    """
    If you want to read the resampling code, please refer to the following path:
        In github resampy library or scipy/signal/_upfirdn_apply.pyx
    """

    sample_rate = 44100
    downsampling = 10000
    upsampling = int(sample_rate * 2.5)

    # test case 1
    test_vector = np.random.randn(sample_rate)
    test_vector_downsample = resample_oct(
        test_vector, downsampling, sample_rate
    )  # FS = 10000
    test_vector_downsample_librosa = librosa.resample(
        y=test_vector, orig_sr=sample_rate, target_sr=downsampling
    )
    test_vector_upsample = resample_oct(test_vector, upsampling, sample_rate)
    test_vector_upsample_librosa = librosa.resample(
        y=test_vector, orig_sr=sample_rate, target_sr=upsampling
    )

    # test case 2
    test_vector = np.zeros(shape=(int(sample_rate * 0.01),))
    total_time = 0.1  # sec
    t = np.arange(0, int(sample_rate * total_time))
    frequency = 1000
    test_vector = np.sin(2 * np.pi * frequency * t / sample_rate)

    test_vector_downsample = resample_oct(test_vector, downsampling, sample_rate)
    test_vector_downsample_librosa = librosa.resample(
        y=test_vector, orig_sr=sample_rate, target_sr=downsampling
    )
    test_vector_upsample = resample_oct(test_vector, upsampling, sample_rate)
    test_vector_upsample_librosa = librosa.resample(
        y=test_vector, orig_sr=sample_rate, target_sr=upsampling
    )


def test_periodogram():
    rng = np.random.default_rng()

    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    freq = 1234.0
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    x = amp * np.sin(2 * np.pi * freq * time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, Pxx_den = signal.welch(x, fs, nperseg=1024)

    plt.semilogy(f, Pxx_den)
    plt.ylim([0.5e-3, 1])
    plt.xlabel("frequency [Hz]")
    plt.ylabel("PSD [V**2/Hz]")
    plt.show()


def test_window():
    sample_rate = 256
    window = signal.windows.gaussian(M=sample_rate, std=0.5)
    window_ = signal.windows.general_gaussian(M=sample_rate, p=1, sig=sample_rate / 10)
    plt.plot(window)
    plt.plot(window_)
    plt.show()


def test_stride():
    # test case 1
    vector = np.arange(0, 7, dtype=np.float64)
    m = _symmetric_toeplitz_numpy(vector)
    print(m)
    from torchmetrics.functional.audio.sdr import _symmetric_toeplitz

    vector_tensor = torch.tensor(vector, dtype=torch.float64)
    m = _symmetric_toeplitz(vector_tensor)
    print(m)

    # test case 2
    vector = np.arange(0, 512, dtype=np.float64)
    m = _symmetric_toeplitz_numpy(vector)
    print(m)

    vector_tensor = torch.tensor(vector, dtype=torch.float64)
    m = _symmetric_toeplitz(vector_tensor)
    print(m)


def test_iir_pesq():
    fs_8k = 8000
    len_sos_8k = 8
    fs_16k = 16000
    len_sos_16k = 12

    sos_8k = [
        0.885535424,
        -0.885535424,
        0.000000000,
        -0.771070709,
        0.000000000,
        0.895092588,
        1.292907193,
        0.449260174,
        1.268869037,
        0.442025372,
        4.049527940,
        -7.865190042,
        3.815662102,
        -1.746859852,
        0.786305963,
        0.500002353,
        -0.500002353,
        0.000000000,
        0.000000000,
        0.000000000,
        0.565002834,
        -0.241585934,
        -0.306009671,
        0.259688659,
        0.249979657,
        2.115237288,
        0.919935084,
        1.141240051,
        -1.587313419,
        0.665935315,
        0.912224584,
        -0.224397719,
        -0.641121413,
        -0.246029464,
        -0.556720590,
        0.444617727,
        -0.307589321,
        0.141638062,
        -0.996391149,
        0.502251622,
    ]
    sos_16k = [
        0.325631521,
        -0.086782860,
        -0.238848661,
        -1.079416490,
        0.434583902,
        0.403961804,
        -0.556985881,
        0.153024077,
        -0.415115835,
        0.696590244,
        4.736162769,
        3.287251046,
        1.753289019,
        -1.859599046,
        0.876284034,
        0.365373469,
        0.000000000,
        0.000000000,
        -0.634626531,
        0.000000000,
        0.884811506,
        0.000000000,
        0.000000000,
        -0.256725271,
        0.141536777,
        0.723593055,
        -1.447186099,
        0.723593044,
        -1.129587469,
        0.657232737,
        1.644910855,
        -1.817280902,
        1.249658063,
        -1.778403899,
        0.801724355,
        0.633692689,
        -0.284644314,
        -0.319789663,
        0.000000000,
        0.000000000,
        1.032763031,
        0.268428979,
        0.602913323,
        0.000000000,
        0.000000000,
        1.001616361,
        -0.823749013,
        0.439731942,
        -0.885778255,
        0.000000000,
        0.752472096,
        -0.375388990,
        0.188977609,
        -0.077258216,
        0.247230734,
        1.023700575,
        0.001661628,
        0.521284240,
        -0.183867259,
        0.354324187,
    ]

    sos_8k = np.array(sos_8k, dtype=np.float64)
    sos_8k = sos_8k.reshape(len_sos_8k, 5)
    sos_8k_full = np.zeros((len_sos_8k, 6), dtype=np.float64)
    sos_8k_full[:, :-3] = sos_8k[:, :-2]
    sos_8k_full[:, -2:] = sos_8k[:, -2:]
    sos_8k_full[:, -3] = 1

    sos_16k = np.array(sos_16k)
    sos_16k = sos_16k.reshape(len_sos_16k, 5)
    sos_16k_full = np.zeros((len_sos_16k, 6), dtype=np.float64)
    sos_16k_full[:, :-3] = sos_16k[:, :-2]
    sos_16k_full[:, -2:] = sos_16k[:, -2:]
    sos_16k_full[:, -3] = 1

    import scipy.signal as signal

    w_8k, h_8k = signal.sosfreqz(sos_8k_full, worN=1024)
    w_16k, h_16k = signal.sosfreqz(sos_16k_full, worN=1024)

    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 8))

    h_8k[np.where(h_8k == 0)] = np.finfo(float).eps
    h_16k[np.where(h_16k == 0)] = np.finfo(float).eps

    axes = [0] * 4
    axes[0] = figure.add_subplot(221)
    axes[1] = figure.add_subplot(222)
    axes[2] = figure.add_subplot(223)
    axes[3] = figure.add_subplot(224)

    axes[0].plot(fs_8k * w_8k / (2 * np.pi), 20 * np.log10(abs(h_8k)))
    axes[0].set_title("8k, Frequency Response")
    axes[1].plot(fs_16k * w_16k / (2 * np.pi), 20 * np.log10(abs(h_16k)))
    axes[1].set_title("16k, Frequency Response")

    for ax in axes[:2]:
        ax.set_xscale("log")
        ax.set_ylabel("Amplitude [dB]")
        ax.set_xlabel("Frequency [rad/sample]")
        ax.grid()

    axes[2].plot(w_8k, np.angle(h_8k))
    axes[2].set_title("8k, Phase Response")
    axes[3].plot(w_16k, np.angle(h_16k))
    axes[3].set_title("16k, Phase Response")

    for ax in axes[2:]:
        ax.set_ylabel("Angle (radians)")
        ax.set_xlabel("Frequency [rad/sample]")
        ax.grid()

    plt.show()


if __name__ == "__main__":
    test_periodogram()
    test_window()
    test_stride()
    test_iir_pesq()

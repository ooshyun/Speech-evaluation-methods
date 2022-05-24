import numpy as np
import scipy.signal as signal


def _estimate_lpc(x):
    """Estimate the LPC, Linear Prediction Code coefficients of a signal
    * When time-delay is 0, auto-correlation has the largest value on the diagonal,
    So it always has inverse matrix
    """
    size_a_coeff = 10
    a_coeff = np.zeros(shape=(size_a_coeff,))
    Rc = np.zeros(shape=(size_a_coeff, size_a_coeff))
    forwardLinearPrediction(Rc, a_coeff, x)

    # Predictive
    # x_predict = np.zeros_like(x)
    # m = a_coeff.shape[0]
    # for i in range(m, x_predict.shape[0]):
    #     x_predict = 0.
    #     for j in range(m):
    #         x_predict -= a_coeff[j] * x[i-1-j]

    return a_coeff, Rc


def _levinson_durbin(M, y):
    """Levinson-Durbin algorithm from scipy"""
    from scipy.linalg import solve_toeplitz, toeplitz

    x = solve_toeplitz((M[:, 0], M[0, :]), y)
    # T = toeplitz(M[:, 0], M[0, :])
    # assert (y == T.dot(x)).all()
    return solve_toeplitz((M[:, 0], M[0, :]), y)


def _auto_correlation_1D(x):
    """Automatic correlation of a frame signal

    Note
    ----
    1. Convolution
    2. Use fft

    Reference
    ---------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html#scipy.signal.correlate
    """
    auto_correlation = signal.correlate(x, x, mode="full", method="auto")
    return auto_correlation


def forwardLinearPrediction(Rc, a_coeff, x):
    N = x.shape[0] - 1
    m = a_coeff.shape[0]

    R = np.zeros(shape=(m + 1,))

    for i in range(m + 1):
        for j in range(N - i + 1):
            R[i] += x[j] * x[j + i]  # R = _auto_correlation_1D(x)[:m+1]

    Ak = np.zeros(shape=(m + 1,))
    Ak[0] = 1

    Ek = R[0]
    for k in range(m):
        lamb = 0.0
        for j in range(k + 1):
            lamb -= Ak[j] * R[k + 1 - j]
        lamb /= np.maximum(1e-6, Ek)

        for n in range((k + 1) // 2 + 1):
            temp = Ak[k + 1 - n] + lamb * Ak[n]
            Ak[n] = Ak[n] + lamb * Ak[k + 1 - n]
            Ak[k + 1 - n] = temp
        Ek *= 1.0 - lamb**2

    R = R[:-1]
    R_full = np.zeros(shape=(2 * m - 1,))
    R_full[R_full.shape[0] // 2 :] = R
    R_full[: R_full.shape[0] // 2] = R[1:][::-1]
    for i in range(m):
        Rc[i, :] = R_full[R_full.shape[0] // 2 - i : R_full.shape[0] // 2 - i + m]

    a_coeff[:] = Ak[1:]


def cal_cepstrum(a_coeff):
    """Calculate the cepstrum from the coefficient of Linear Predictive Coding"""
    num_p = a_coeff.shape[0]
    cep = np.zeros_like(a_coeff)
    for m in range(num_p):
        a_m = a_coeff[m]
        if m > 1:
            for k in range(m - 1):
                a_m += k / m * cep[k] * a_coeff[m - k]
        cep[m] = a_m
    return cep


def weighted_13_25_band(sampling_rate, nfft, num_bands=25):
    """ """
    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    cent_freq = np.array(
        [
            50.0000,
            120.000,
            190.000,
            260.000,
            330.000,
            400.000,
            470.000,
            540.000,
            617.372,
            703.378,
            798.717,
            904.128,
            1020.38,
            1148.30,
            1288.72,
            1442.54,
            1610.70,
            1794.16,
            1993.93,
            2211.08,
            2446.71,
            2701.97,
            2978.04,
            3276.17,
            3597.63,
        ]
    )

    bandwidth = np.array(
        [
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            70.0000,
            77.3724,
            86.0056,
            95.3398,
            105.411,
            116.256,
            127.914,
            140.423,
            153.823,
            168.154,
            183.457,
            199.776,
            217.153,
            235.631,
            255.255,
            276.072,
            298.126,
            321.465,
            346.136,
        ]
    )

    # articulation index weights
    W = np.array(
        [
            0.003,
            0.003,
            0.003,
            0.007,
            0.010,
            0.016,
            0.016,
            0.017,
            0.017,
            0.022,
            0.027,
            0.028,
            0.030,
            0.032,
            0.034,
            0.035,
            0.037,
            0.036,
            0.036,
            0.033,
            0.030,
            0.029,
            0.027,
            0.026,
            0.026,
        ]
    )

    max_freq = sampling_rate // 2
    _nfft = nfft // 2 + 1
    min_factor = np.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter

    if num_bands == 25:
        sumW = np.sum(W)
        bw_min = bandwidth[0]
        crit_filter = np.zeros(shape=(len(cent_freq), _nfft))

        num_crit = len(cent_freq)
        for i in range(0, num_crit):
            f0 = (cent_freq[i] / max_freq) * (_nfft)
            bw = (bandwidth[i] / max_freq) * (_nfft)
            norm_factor = np.log(bw_min) - np.log(bandwidth[i])

            id_fft = np.arange(0, _nfft)
            crit_filter[i, :] = np.exp(
                -11 * ((np.square((id_fft - np.floor(f0)) / bw))) + norm_factor
            )
            crit_filter[i, :] = crit_filter[i, :] * [crit_filter[i, :] > min_factor]

    elif num_bands == 13:  # use 13 bands
        cent_freq_13band = np.zeros(shape=(13,))
        bandwidth_13band = np.zeros(shape=(13,))
        W2 = np.zeros(shape=(13,))

        # lump adjacent filters together
        k = 2
        cent_freq_13band[0] = cent_freq[0]
        bandwidth_13band[0] = bandwidth[0] + bandwidth[1]
        W2[0] = W[0]
        for i in range(1, 13):
            cent_freq_13band[i] = cent_freq_13band[i - 1] + bandwidth_13band[i - 1]
            bandwidth_13band[i] = bandwidth[k] + bandwidth[k + 1]
            W2[i] = 0.5 * [W[k] + W[k + 1]]
            k = k + 2
        sumW = np.sum(W2)
        bw_min = bandwidth_13band[0]  # minimum critical bandwidth

        crit_filter = np.zeros(shape=(len(cent_freq_13band), _nfft))
        num_crit = len(cent_freq_13band)
        for i in range(0, num_crit):
            f0 = (cent_freq_13band[i] / max_freq) * (_nfft)
            bw = (bandwidth_13band(i) / max_freq) * (_nfft)
            norm_factor = np.log(bw_min) - np.log(bandwidth_13band[i])

            id_fft = np.arange(0, _nfft)
            crit_filter[i, :] = np.exp(
                -11 * ((np.square((id_fft - np.floor(f0)) / bw))) + norm_factor
            )
            crit_filter[i, :] = crit_filter[i, :] * [crit_filter[i, :] > min_factor]

    return crit_filter


def thirdoct(fs, nfft, num_bands, min_freq, window="square"):
    """Returns the 1/3 octave band matrix and its center frequencies

    Paramters
    ---------
        fs : sampling rate
        nfft : FFT size
        num_bands : number of 1/3 octave bands
        min_freq : center frequency of the lowest 1/3 octave band
    Returns
    -------
        obm : Octave Band Matrix
        cf : center frequencies

    TODO LIST
    ---------
        1. Test Gaussian window
            Gaussian Windowed Chirps: https://ccrma.stanford.edu/~jos/sasp/Gaussian_Windowed_Chirps_Chirplets.html
    """
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))  # a verifier

    # square window @ S.O
    for i in range(len(cf)):
        # Match 1/3 oct band freq with fft frequency bin
        f_bin = np.argmin(
            np.square(f - freq_low[i])
        )  # the shortest distance between f and freq_low[i] @ S.O
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(
            np.square(f - freq_high[i])
        )  # the shortest distance between f and freq_high[i] @ S.O
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        # Assign to the octave band matrix
        if window == "square":
            obm[i, fl_ii:fh_ii] = 1
        elif window == "gaussian":
            obm[i, fl_ii:fh_ii] = signal.windows.general_gaussian(
                M=fh_ii - fl_ii, p=1, sig=fh_ii - fl_ii / 10
            )
        else:
            pass

    return obm, cf

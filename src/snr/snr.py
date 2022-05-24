import numpy as np
from .utils import (
    thirdoct,
    cal_cepstrum,
    _estimate_lpc,
    weighted_13_25_band,
)

eps = np.finfo(float).eps


def estimate_snr(clean_signal, enhanced_signal, sampling_rate, zero_mean=False):
    """Signal to Noise Ratio (SNR)

    SNR_{dB} = log_{10} \dfrac{P_{Signal}}{P_{Noise}}
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )

    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    if zero_mean:
        clean_signal = clean_signal - np.mean(clean_signal)
        enhanced_signal = enhanced_signal - np.mean(enhanced_signal)

    p_clean_signal = np.sum(np.power(clean_signal - np.mean(clean_signal), 2))
    # Clean_signal energy

    noise_signal = enhanced_signal - clean_signal  # Noise signal
    p_noise_signal = np.sum(np.power(noise_signal - np.mean(noise_signal), 2))

    snr = 10 * np.log10(p_clean_signal / p_noise_signal)
    return snr


def estimate_segsnr(
    clean_signal,
    enhanced_signal,
    sampling_rate,
    window_size=1024,
    hop=256,
    upgrade=False,
):
    """Segmental Signal to Noise Ratio (segSNR)

    Reference
    ---------
    - Hansen, John HL, and Bryan L. Pellom. "An effective quality evaluation protocol for speech enhancement algorithms." ICSLP. Vol. 7. 1998.
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    - https://github.com/IMLHF/Speech-Enhancement-Measures
    - https://github.com/schmiph2/pysepm
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )
    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    MIN_SNR = -20 / 10
    MAX_SNR = 35 / 10

    window = np.hanning(window_size)

    noise_signal = enhanced_signal - clean_signal  # Noise signal

    clean_signal_frames = np.array(
        [
            window * clean_signal[i : i + window_size]
            for i in range(0, len(clean_signal) - window_size, hop)
        ]
    )
    noise_signal_frames = np.array(
        [
            window * noise_signal[i : i + window_size]
            for i in range(0, len(noise_signal) - window_size, hop)
        ]
    )

    segsnr = np.zeros(clean_signal_frames.shape[0])
    bufsnr = 0
    for iframe in range(clean_signal_frames.shape[0]):
        clean_signal_frame_energy = np.sum(np.power(clean_signal_frames[iframe, :], 2))
        noise_signal_frame_energy = np.sum(np.power(noise_signal_frames[iframe, :], 2))
        if upgrade:
            bufsnr = np.log10(1 + clean_signal_frame_energy / noise_signal_frame_energy)
        else:
            bufsnr = np.log10(clean_signal_frame_energy / noise_signal_frame_energy)

        if bufsnr > MAX_SNR:
            bufsnr = MAX_SNR
        if bufsnr < MIN_SNR:
            bufsnr = MIN_SNR

        segsnr[iframe] = bufsnr

    segsnr = segsnr[:-1]  # Remove the last one

    return 10 * np.mean(segsnr)


def estimate_llr(
    clean_signal,
    enhanced_signal,
    sampling_rate,
    window_size=1024,
    hop=256,
    used_for_composite=False,
):
    """Log-Likelihood Ratio (LLR)

    Reference
    ---------
    - S. Quackenbush, T. Barnwell, and M. Clements, Objective Measures of Speech Quality. Englewood Cliffs, NJ: Prentice-Hall, 1988.
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    - https://github.com/IMLHF/Speech-Enhancement-Measures
    - https://github.com/schmiph2/pysepm
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )
    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    window = np.hanning(window_size)

    clean_signal_frames = np.array(
        [
            window * clean_signal[i : i + window_size]
            for i in range(0, len(clean_signal) - window_size, hop)
        ]
    )
    enhanced_signal_frames = np.array(
        [
            window * enhanced_signal[i : i + window_size]
            for i in range(0, len(enhanced_signal) - window_size, hop)
        ]
    )

    d_llr = np.zeros(shape=(clean_signal_frames.shape[0],))
    for iframe in range(clean_signal_frames.shape[0]):
        lpc_vec_clean, R_m_clean = _estimate_lpc(clean_signal_frames[iframe, :])
        lpc_vec_enhance, _ = _estimate_lpc(enhanced_signal_frames[iframe, :])
        lpc_vec_clean = np.expand_dims(lpc_vec_clean, axis=-1)
        lpc_vec_enhance = np.expand_dims(lpc_vec_enhance, axis=-1)

        _d_llr = (lpc_vec_enhance.T @ R_m_clean @ lpc_vec_enhance) / (
            lpc_vec_clean.T @ R_m_clean @ lpc_vec_clean
        )
        _d_llr[np.isnan(_d_llr)] = np.inf
        _d_llr[np.where(_d_llr == 0)] = eps
        _d_llr = np.log(_d_llr)

        # The segmental LLR values were limited in the range of [0, 2]
        # to further reduce the number of outliers.
        if 0 < _d_llr <= 2:
            d_llr[iframe] = _d_llr
        elif _d_llr <= 0:
            pass
        elif _d_llr > 2:
            if not used_for_composite:
                d_llr[iframe] = 2
            else:
                d_llr[iframe] = _d_llr
        else:
            pass

    # smallest 95% of the frame LLR values were used to compute the average LLR value
    d_llr = np.sort(d_llr, axis=0)
    alpha = 0.95
    if 0 in d_llr:
        izero = np.where(d_llr == 0)[0][-1]
        d_llr = d_llr[izero + 1 : izero + 1 + int((len(d_llr) - izero + 1) * alpha) + 1]
    else:
        d_llr = d_llr[: int(len(d_llr) * alpha) + 1]

    d_llr = np.mean(d_llr)

    return d_llr


def estimate_is(
    clean_signal, enhanced_signal, sampling_rate, window_size=1024, hop=256
):
    """Itakura-Saito distance measure (IS)

    Reference
    ---------
    - S. Quackenbush, T. Barnwell, and M. Clements, Objective Measures of Speech Quality. Englewood Cliffs, NJ: Prentice-Hall, 1988.
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    - https://github.com/IMLHF/Speech-Enhancement-Measures
    - https://github.com/schmiph2/pysepm
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )
    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    window = np.hanning(window_size)

    clean_signal_frames = np.array(
        [
            window * clean_signal[i : i + window_size]
            for i in range(0, len(clean_signal) - window_size, hop)
        ]
    )
    enhanced_signal_frames = np.array(
        [
            window * enhanced_signal[i : i + window_size]
            for i in range(0, len(enhanced_signal) - window_size, hop)
        ]
    )

    d_is = np.zeros(shape=(clean_signal_frames.shape[0],))
    for iframe in range(clean_signal_frames.shape[0]):
        lpc_vec_clean, R_m_clean = _estimate_lpc(clean_signal_frames[iframe, :])
        lpc_vec_enhance, _ = _estimate_lpc(enhanced_signal_frames[iframe, :])

        gain_lpc_vec_clean = np.abs(1 / (1 - np.sum(lpc_vec_clean))) ** 2
        gain_lpc_vec_enhance = np.abs(1 / (1 - np.sum(lpc_vec_enhance))) ** 2
        gain_ratio = gain_lpc_vec_clean / gain_lpc_vec_enhance

        lpc_vec_clean = np.expand_dims(lpc_vec_clean, axis=-1)
        lpc_vec_enhance = np.expand_dims(lpc_vec_enhance, axis=-1)

        _d_is = (
            gain_ratio
            * lpc_vec_enhance.T
            @ R_m_clean
            @ lpc_vec_enhance
            / lpc_vec_clean.T
            @ R_m_clean
            @ lpc_vec_clean
            + np.log(gain_ratio)
            - 1
        )

        # The segmental LLR values were limited in the range of [0, 100]
        # to further reduce the number of outliers.
        if 0 < _d_is <= 100:
            d_is[iframe] = _d_is
        elif _d_is == 0:
            d_is[iframe] = eps
        else:
            pass

    d_is = np.sort(d_is, axis=0)
    if 0 in d_is:
        izero = np.where(d_is == 0)[0][-1]
        d_is = d_is[izero + 1 :]
    d_is = np.average(d_is)

    return d_is


def estimate_cep(
    clean_signal, enhanced_signal, sampling_rate, window_size=1024, hop=256
):
    """Cepstrum distance measures (CEP)

    Reference
    ---------
    - S. Quackenbush, T. Barnwell, and M. Clements, Objective Measures of Speech Quality. Englewood Cliffs, NJ: Prentice-Hall, 1988.
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    - https://github.com/IMLHF/Speech-Enhancement-Measures
    - https://github.com/schmiph2/pysepm
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )
    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    window = np.hanning(window_size)
    clean_signal_frames = np.array(
        [
            window * clean_signal[i : i + window_size]
            for i in range(0, len(clean_signal) - window_size, hop)
        ]
    )
    enhanced_signal_frames = np.array(
        [
            window * enhanced_signal[i : i + window_size]
            for i in range(0, len(enhanced_signal) - window_size, hop)
        ]
    )

    d_cep = np.zeros(shape=(clean_signal_frames.shape[0],))
    for iframe in range(clean_signal_frames.shape[0]):
        lpc_vec_clean, R_m_clean = _estimate_lpc(clean_signal_frames[iframe, :])
        lpc_vec_enhance, _ = _estimate_lpc(enhanced_signal_frames[iframe, :])

        cep_clean = cal_cepstrum(lpc_vec_clean)
        cep_enhance = cal_cepstrum(lpc_vec_enhance)

        _d_cep = 10 / np.log(10) * np.sqrt(2 * np.sum((cep_clean - cep_enhance) ** 2))
        if 0 < _d_cep <= 10:
            d_cep[iframe] = _d_cep
        elif _d_cep == 0:
            d_cep[iframe] = eps
        else:
            pass

    d_cep = np.sort(d_cep, axis=0)
    if 0 in d_cep:
        izero = np.where(d_cep == 0)[0][-1]
        d_cep = d_cep[izero + 1 :]
    d_cep = np.average(d_cep)

    return d_cep


def estimate_fwssnr(
    clean_signal, enhanced_signal, sampling_rate, window_size=1024, hop=256
):
    """Frequency-weighted segmental SNR (fwSNRseg)

    Reference
    ---------
    - Tribolet, J., Noll, P., McDermott, B., Crochiere, R.E.: A study of complexity and quality of speech waveform coders.
    In: Proc. IEEE Int. Conf. Acoust. Speech, Signal Processing, pp. 586-590 (1978)
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    - https://github.com/IMLHF/Speech-Enhancement-Measures
    - https://github.com/schmiph2/pysepm
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )
    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    window = np.hanning(window_size)
    clean_signal_frames = np.array(
        [
            window * clean_signal[i : i + window_size]
            for i in range(0, len(clean_signal) - window_size, hop)
        ]
    )
    enhanced_signal_frames = np.array(
        [
            window * enhanced_signal[i : i + window_size]
            for i in range(0, len(enhanced_signal) - window_size, hop)
        ]
    )

    fwSNRseg = np.zeros(shape=(clean_signal_frames.shape[0],))
    OCTAVE_METHOD = False

    for iframe in range(clean_signal_frames.shape[0]):
        nfft = window_size * 2
        # fft including normalized
        clean_signal_frame_fft = (
            np.fft.fft(clean_signal_frames[iframe, :], n=nfft) / nfft
        )
        enhanced_signal_frame_fft = (
            np.fft.fft(enhanced_signal_frames[iframe, :], n=nfft) / nfft
        )
        clean_signal_frame_fft = clean_signal_frame_fft[: nfft // 2 + 1]
        enhanced_signal_frame_fft = enhanced_signal_frame_fft[: nfft // 2 + 1]

        clean_signal_frame_fft = np.abs(clean_signal_frame_fft)
        enhanced_signal_frame_fft = np.abs(enhanced_signal_frame_fft)

        # Normalized
        clean_signal_frame_fft = clean_signal_frame_fft / np.sum(clean_signal_frame_fft)
        enhanced_signal_frame_fft = enhanced_signal_frame_fft / np.sum(
            enhanced_signal_frame_fft
        )

        # Paper method
        num_bands = 25
        weighted_window = weighted_13_25_band(sampling_rate, nfft, num_bands=num_bands)
        clean_band_abs = np.matmul(weighted_window, clean_signal_frame_fft)
        enhanced_band_abs = np.matmul(weighted_window, enhanced_signal_frame_fft)

        if OCTAVE_METHOD:  # octave method not in paper
            min_freq = 50
            num_bands = 27
            # In proportion to the ear's critical bands, Octave 1/3
            # 15 -> min frequency 150Hz, 13 bands
            # 27 -> min frequency  50Hz, 25 bands
            octave_band_matrix, cf = thirdoct(
                sampling_rate, nfft, num_bands, min_freq, window="gaussian"
            )  # square, gaussian based on pystoi
            # Apply Octave band matrix to the spectrograms
            clean_band_abs = np.sqrt(
                np.matmul(octave_band_matrix, np.square(clean_signal_frame_fft))
            )
            enhanced_band_abs = np.sqrt(
                np.matmul(octave_band_matrix, np.square(enhanced_signal_frame_fft))
            )

        gamma = 0.2  # 0.1 - 2 in papers
        weight_clean_band = np.power(clean_band_abs, gamma)

        denominator = np.square(clean_band_abs - enhanced_band_abs)
        id_out_of_band = np.where(denominator == 0)
        log_fwSNRseg = 10 * np.log10(np.square(clean_band_abs) / denominator)
        log_fwSNRseg[id_out_of_band] = 0

        _fwSNRseg = np.sum(weight_clean_band * log_fwSNRseg) / np.sum(weight_clean_band)

        fwSNRseg[iframe] = min(max(_fwSNRseg, -10), 35)

    fwSNRseg = np.sum(fwSNRseg) / clean_signal_frames.shape[0]
    return fwSNRseg


def estimate_wss(
    clean_signal, enhanced_signal, sampling_rate, window_size=1024, hop=256
):
    """Weighted spectral slope (WSS)

    Reference
    ---------
    - Klatt, D.H.: Prediction of perceived phonetic distance from critical-band spectra: A first step.
    In: Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing, vol. 2, pp. 1278-1281 (1982)
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    - https://github.com/IMLHF/Speech-Enhancement-Measures
    - https://github.com/schmiph2/pysepm
    """
    size = (
        len(enhanced_signal)
        if len(clean_signal) > len(enhanced_signal)
        else len(clean_signal)
    )
    clean_signal = clean_signal[:size]
    enhanced_signal = enhanced_signal[:size]

    window = np.hanning(window_size)
    clean_signal_frames = np.array(
        [
            window * clean_signal[i : i + window_size]
            for i in range(0, len(clean_signal) - window_size, hop)
        ]
    )
    enhanced_signal_frames = np.array(
        [
            window * enhanced_signal[i : i + window_size]
            for i in range(0, len(enhanced_signal) - window_size, hop)
        ]
    )

    wss = np.zeros(shape=(clean_signal_frames.shape[0],))
    Kmax = 20
    # value suggested by Klatt, pg 1280
    Klocmax = 1
    # value suggested by Klatt, pg 1280

    for iframe in range(clean_signal_frames.shape[0]):
        nfft = window_size * 2
        # fft including normalized
        clean_signal_frame_fft = (
            np.fft.fft(clean_signal_frames[iframe, :], n=nfft) / nfft
        )
        enhanced_signal_frame_fft = (
            np.fft.fft(enhanced_signal_frames[iframe, :], n=nfft) / nfft
        )
        clean_signal_frame_fft = clean_signal_frame_fft[: nfft // 2 + 1]
        enhanced_signal_frame_fft = enhanced_signal_frame_fft[: nfft // 2 + 1]

        """In this point, it is possible to use LPC vector instead of energy"""
        clean_signal_frame_fft = np.square(np.abs(clean_signal_frame_fft))
        enhanced_signal_frame_fft = np.square(np.abs(enhanced_signal_frame_fft))

        num_bands = 25
        weighted_window = weighted_13_25_band(sampling_rate, nfft, num_bands=num_bands)
        clean_band_abs = np.matmul(weighted_window, clean_signal_frame_fft)
        enhanced_band_abs = np.matmul(weighted_window, enhanced_signal_frame_fft)

        """TODO: Exception log10(0)"""
        clean_energy = 10 * np.log10(clean_band_abs)
        enhanced_energy = 10 * np.log10(enhanced_band_abs)

        # Compute Spectral Slope (dB[i+1]-dB[i])
        clean_slope = np.diff(clean_energy)
        enhanced_slope = np.diff(enhanced_energy)

        # Find the nearest peak locations in the spectra to
        # each critical band.  If the slope is negative, we
        # search to the left.  If positive, we search to the
        # right.
        clean_peak_locations = np.zeros(shape=(num_bands - 1,))
        enhanced_peak_locations = np.zeros(shape=(num_bands - 1,))

        for iband in range(num_bands - 1):
            if clean_slope[iband] > 0:
                n = iband
                while n < num_bands and clean_slope[n] > 0:
                    n += 1
                clean_peak_locations[iband] = clean_energy[n - 1]
            else:
                n = iband
                while n > 0 and clean_slope[n] <= 0:
                    n -= 1
                clean_peak_locations[iband] = clean_energy[n + 1]

            if enhanced_slope[iband] > 0:
                n = iband
                while n < num_bands and enhanced_slope[n] > 0:
                    n += 1
                enhanced_peak_locations[iband] = enhanced_energy[n - 1]
            else:
                n = iband
                while n > 0 and enhanced_slope[n] <= 0:
                    n -= 1
                enhanced_peak_locations[iband] = enhanced_energy[n + 1]

        # Compute the WSS Measure
        dBMax_clean = np.max(clean_energy)
        dBMax_enhanced = np.max(enhanced_energy)

        Wmax_clean = Kmax / (Kmax + dBMax_clean - clean_energy[:-1])
        Wlocmax_clean = Klocmax / (Klocmax + clean_peak_locations - clean_energy[:-1])
        W_clean = Wmax_clean * Wlocmax_clean

        Wmax_enhanced = Kmax / (Kmax + dBMax_enhanced - enhanced_energy[:-1])
        Wlocmax_enhanced = Klocmax / (
            Klocmax + enhanced_peak_locations - enhanced_energy[:-1]
        )
        W_enhanced = Wmax_enhanced * Wlocmax_enhanced

        W = (W_clean + W_enhanced) / 2

        wss[iframe] = np.sum(W * np.square(clean_slope - enhanced_slope))
        wss[iframe] = wss[iframe] / np.sum(W)

    wss = np.sum(wss) / clean_signal_frames.shape[0]
    return wss


def estimate_si_snr(clean_signal, enhanced_signal, sampling_rate, zero_mean=True):
    """Scale-Invariant Source-to-Noise Ratio(SI-SNR)
    or Scale-Invariant Signal Distortion Ratio(SI-SDR)

    References
    ----------
    - Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
    and Signal Processing (ICASSP) 2019.
    - https://github.com/PyTorchLightning/metrics
    """
    if zero_mean:
        clean_signal = clean_signal - np.mean(clean_signal, axis=-1, keepdims=True)
        enhanced_signal = enhanced_signal - np.mean(
            enhanced_signal, axis=-1, keepdims=True
        )

    alpha = (np.sum(enhanced_signal * clean_signal, axis=-1, keepdims=True) + eps) / (
        np.sum(clean_signal**2, axis=-1, keepdims=True) + eps
    )
    projection = alpha * clean_signal

    noise = enhanced_signal - projection

    ratio = (np.sum(projection**2, axis=-1) + eps) / (
        np.sum(noise**2, axis=-1) + eps
    )
    ratio = 10 * np.log10(ratio)

    return ratio

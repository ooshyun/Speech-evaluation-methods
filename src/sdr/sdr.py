import musdb
import museval
import numpy as np
from numpy.linalg import norm
import scipy.linalg as linalg

eps = np.finfo(dtype=np.float64).eps

from .util import _compute_autocorr_crosscorr_numpy, _symmetric_toeplitz_numpy

toeplitz_conjugate_gradient = None  # fast bss is not implemented
_FAST_BSS_EVAL_AVAILABLE = False  # fast bss is not implemented


def estimate_sdr(clean_signal, enhanced_signal, sampling_rate):
    """Signal to Distortion Ratio (SDR) from museval

    Reference
    ---------
    - https://github.com/sigsep/sigsep-mus-eval
    - Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
    IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462-1469.

    """
    sdr, isr, sir, sar, perm = museval.metrics.bss_eval(clean_signal, enhanced_signal)
    return sdr


def estimate_sdr_scratch(
    clean_signal,
    enhanced_signal,
    sampling_rate,
    filter_length=512,
    zero_mean=True,
    use_cg_iter=None,
    load_diag=None,
):
    """Signal to Distortion Ratio (SDR)

    This function is from pytorchlightening/metrics and convert to numpy for understanding.

    Reference
    ---------
    - https://github.com/PyTorchLightning/metrics
    - Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
    IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462-1469.
    - Scheibler, R. (2021). SDR -- Medium Rare with Fast Computations.
    """
    # use double precision
    enhanced_signal_dtype = enhanced_signal.dtype
    enhanced_signal = enhanced_signal.astype(np.double)
    clean_signal = clean_signal.astype(np.double)

    if zero_mean:
        enhanced_signal = enhanced_signal - np.mean(
            enhanced_signal, axis=-1, keepdims=True
        )
        clean_signal = clean_signal - np.mean(clean_signal, axis=-1, keepdims=True)

    # normalize along time-axis to make preds and target have unit norm
    enhanced_signal = enhanced_signal / np.clip(
        norm(enhanced_signal, axis=-1, keepdims=True), a_min=1e-6, a_max=None
    )
    clean_signal = clean_signal / np.clip(
        norm(clean_signal, axis=-1, keepdims=True), a_min=1e-6, a_max=None
    )  # @ S.O min, max cliping

    # solve for the optimal filter
    # compute auto-correlation and cross-correlation
    r_0, b = _compute_autocorr_crosscorr_numpy(
        clean_signal, enhanced_signal, corr_len=filter_length
    )  # @ S.O r_0 auto-correlation, b cross-correlation

    if load_diag is not None:
        # the diagonal factor of the Toeplitz matrix is the first coefficient of r_0
        r_0[..., 0] += load_diag

    if use_cg_iter is not None and _FAST_BSS_EVAL_AVAILABLE:
        # use preconditioned conjugate gradient
        sol = toeplitz_conjugate_gradient(
            r_0, b, n_iter=use_cg_iter
        )  # @ S.O faster way, [2]
    else:
        # regular matrix solver
        r = _symmetric_toeplitz_numpy(
            r_0
        )  # the auto-correlation of the L shifts of `target`
        try:
            sol = linalg.solve(r, b)
        except np.linalg.LinAlgError:
            print("Warning: The matrix is LinAlgError")
            sol = linalg.lstsq(r, b)[0]
        # @ S.O solve AX = B , which is similar to getting the filter coefficients in Linear Predictive Coding
        # @ S.O but the shape of A and B is differenct, it is Projection

    # compute the coherence
    coh = np.einsum("...l,...l->...", b, sol)  # @ S.O coherence = convolve(b * sol)

    # transform to decibels
    ratio = coh / (1 - coh)
    val = 10.0 * np.log10(ratio)

    if enhanced_signal_dtype == np.float64:
        return val
    else:
        return val.astype(np.float64)


def estimate_sdr_several_reference(data_path, index_track=0):
    """Signal to Distortion Ratio (SDR)

    Reference
    ---------
    [1] Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
    IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462-1469.

    [2] Scheibler, R. (2021). SDR -- Medium Rare with Fast Computations.

    Evaluate several sources and channels on the same track.
    This simple version to compare source and estimated signals is si-sdr(or si-snr)
    """

    def estimate_and_evaluate(track):
        # assume mix as estimates
        estimates = {"vocals": track.audio, "accompaniment": track.audio}

        # Evaluate using museval
        scores = museval.eval_mus_track(track, estimates, output_dir="./data/result/")
        # print nicely formatted and aggregated scores
        print(scores)

    mus = musdb.DB(root=data_path)
    print(data_path)
    if len(mus.tracks) == 0:
        print("Warning: No track found in directory.")
    else:
        for itrack, track in enumerate(mus):
            if itrack == index_track:
                estimate_and_evaluate(track)


def estimate_si_sdr(
    clean_signal, enhanced_signal, sampling_rate, zero_mean: bool = True
):
    """Scale-Invariant Source-to-Noise Ratio(SI-SNR)
    or Scale-Invariant Signal Distortion Ratio(SI-SDR)

    References
    ----------
    - https://github.com/PyTorchLightning/metrics
    - Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE International Conference on Acoustics, Speech
    and Signal Processing (ICASSP) 2019.
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

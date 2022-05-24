import numpy as np
from resampy import resample
from .snr import estimate_wss, estimate_llr, estimate_segsnr
from .pesq import estimate_pesq


def estimate_composite(clean_speech, enhanced_signal, sampling_rate, mode):
    """Signal distortion(SIG), Background intrusiveness(BAK), The Overall Quality of Speech(OVRL)

    Notes
    -----
    The speech signal alone using a five-point scale of signal distortion (SIG)
    The background noise alone using a five-point scale of background intrusiveness (BAK)
    The overall quality using the scale of the mean opinion score (OVRL)
    1=bad 2=poor 3=fair 4=good 5=excellent

    Reference
    ---------
    - https://github.com/schmiph2/pysepm
    - Speech Enhancement Theory and Practice, Second Edition, Philipos C. Loizou.
    """
    wss_dist = estimate_wss(clean_speech, enhanced_signal, sampling_rate)
    llr_mean = estimate_llr(
        clean_speech, enhanced_signal, sampling_rate, used_for_composite=True
    )
    segSNR = estimate_segsnr(clean_speech, enhanced_signal, sampling_rate)
    pesq_mos, mos_lqo = estimate_pesq(
        clean_speech, enhanced_signal, sampling_rate, mode=mode
    )

    if sampling_rate >= 16e3:
        used_pesq_val = mos_lqo
    else:
        used_pesq_val = pesq_mos

    sig = 3.093 - 1.029 * llr_mean + 0.603 * used_pesq_val - 0.009 * wss_dist
    sig = np.max((1, sig))
    sig = np.min((5, sig))  # limit values to [1, 5]

    bak = 1.634 + 0.478 * used_pesq_val - 0.007 * wss_dist + 0.063 * segSNR
    bak = np.max((1, bak))
    bak = np.min((5, bak))  # limit values to [1, 5]

    ovrl = 1.594 + 0.805 * used_pesq_val - 0.512 * llr_mean - 0.007 * wss_dist
    ovrl = np.max((1, ovrl))
    ovrl = np.min((5, ovrl))  # limit values to [1, 5]

    return sig, bak, ovrl

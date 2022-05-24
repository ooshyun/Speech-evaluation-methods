import pesq
import numpy as np


def estimate_pesq(clean_speech, enhanced_speech, sampling_rate, mode):
    """Perceptual evaluation of speech quality (PESQ)

    Reference
    ---------
    https://github.com/ludlows/python-pesq
    """
    mos_lqo = pesq.pesq(sampling_rate, clean_speech, enhanced_speech, mode)
    if sampling_rate == 8000:
        pesq_mos = (
            46607 / 14945 - (2000 * np.log(1 / (mos_lqo / 4 - 999 / 4000) - 1)) / 2989
        )  # 0.999 + ( 4.999-0.999 ) / ( 1+np.exp(-1.4945*pesq_mos+4.6607) )
    elif sampling_rate == 16000:
        pesq_mos = np.NaN
    else:
        raise ValueError("sampling rate must be either 8 kHz or 16 kHz")

    return pesq_mos, mos_lqo

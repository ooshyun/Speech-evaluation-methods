"""Speech Intelligibility (STOI)

    Note
    ----
    In short, the correlation between processed speech and clean speech based on Octave band (150-5KHz), 
    considering SDR(Signal Distortion Ratio) and size of STFT frame. 

    Reference
    ---------
    - https://github.com/mpariente/pystoi/tree/resampling_checks
    - Taal, Cees H., et al. "A short-time objective intelligibility measure for time-frequency weighted noisy speech.", 
        2010 IEEE international conference on acoustics, speech and signal processing. IEEE, 2010.

"""
from pystoi import stoi as estimate_stoi
from pystoi.utils import resample_oct

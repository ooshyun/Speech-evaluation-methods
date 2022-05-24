import numpy as np
import math


def _as_strided_scratch(a, size, stride, storage_offset=0):
    result = np.zeros(shape=size, dtype=a.dtype)

    nrow = size[-2]
    ncol = size[-1]

    row_stride = stride[-2]
    col_stride = stride[-1]

    origin = 0
    for col in range(ncol):
        for row in range(nrow):
            result[..., col, row] = a[..., origin + row * row_stride]
        origin += col_stride

    return result


def _compute_autocorr_crosscorr_numpy(target, preds, corr_len: int):
    """This api is from pytorchlightening/metrics and convert to numpy for understanding

    Below explanation is same as pytorch.

    Parameters
    ----------
    target: the target (reference) signal of shape [..., time]
    preds: the preds (estimated) signal of shape [..., time]
    corr_len: the length of the auto correlation and cross correlation

    Returns
    -------
    the auto correlation of `target` of shape [..., corr_len]
    the cross correlation of `target` and `preds` of shape [..., corr_len]

    Notes
    -----
    Compute the auto correlation of `target` and the cross correlation of `target` and `preds` using the fast
    Fourier transform (FFT). Let's denotes the symmetric Toeplitz matric of the auto correlation of `target` as
    `R`, the cross correlation as 'b', then solving the equation `Rh=b` could have `h` as the coordinate of
    `preds` in the column space of the `corr_len` shifts of `target`.
    """
    # the valid length for the signal after convolution
    n_fft = 2 ** math.ceil(
        math.log2(preds.shape[-1] + target.shape[-1] - 1)
    )  # @ S.O Q1

    # computes the auto correlation of `target`
    # r_0 is the first row of the symmetric Toeplitz matric
    t_fft = np.fft.rfft(target, n=n_fft, axis=-1)
    r_0 = np.fft.irfft(t_fft.real**2 + t_fft.imag**2, n=n_fft)[
        ..., :corr_len
    ]  # @ S.O Q2

    # computes the cross-correlation of `target` and `preds`
    p_fft = np.fft.rfft(preds, n=n_fft, axis=-1)
    b = np.fft.irfft(t_fft.conj() * p_fft, n=n_fft, axis=-1)[..., :corr_len]  # @ S.O Q4

    return r_0, b


def _symmetric_toeplitz_numpy(vector):
    """This api is from pytorchlightening/metrics and convert to numpy for understanding
    Below explanation is same as pytorch.
    """
    # @ S.O Why vector[..., 1:], not vector[1:]?
    # It's designed to mean at this point, insert as many full slices (:) to extend the multi-dimensional slice to all dimensions.
    # a = arange(16).reshape(2,2,2,2) -> a[..., 0].flatten() = a[:,:,:,0].flatten()
    vec_exp = np.concatenate([np.flip(vector, axis=(-1,)), vector[..., 1:]], axis=-1)
    v_len = vector.shape[-1]

    # @ S.O vec_exp.shape[:-1] if there's several source or channel, then its shape will be (..., v_len)
    # It will be itertated utill v_len

    # @ S.O numpy as_strided is overflowed when trying float64
    return np.flip(
        _as_strided_scratch(
            vec_exp,
            size=vec_exp.shape[:-1] + (v_len, v_len),
            stride=vec_exp.strides[:-1] + (1, 1),
        ),
        axis=-1,
    )

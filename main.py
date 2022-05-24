import os
import torch
import torchaudio
import torchmetrics
import soundfile as sf

from src import (
    estimate_snr,
    estimate_segsnr,
    estimate_llr,
    estimate_is,
    estimate_cep,
    estimate_fwssnr,
    estimate_wss,
    estimate_sdr,
    estimate_sdr_scratch,
    estimate_sdr_several_reference,
    estimate_si_snr,
    estimate_si_sdr,
    estimate_stoi,
    estimate_pesq,
    estimate_composite,
)

# PATH_CLEAN_SPEECH = ""                  # path/to/clean/audio
# PATH_ENHANCED_SPEECH = ""               # path/to/denoised/audio

PATH_CLEAN_SPEECH = "./data/wav/clean/sp01.wav"
PATH_ENHANCED_SPEECH = "./data/wav/enhance/sp01_babble_sn10.wav"

clean_speech, fs = sf.read(PATH_CLEAN_SPEECH)
enhanced_speech, fs = sf.read(PATH_ENHANCED_SPEECH)

clean_speech_torch, fs = torchaudio.load(PATH_CLEAN_SPEECH)
enhanced_speech_torch, fs = torchaudio.load(PATH_ENHANCED_SPEECH)


def test_snr():
    """Signal to Noise Ratio (SNR)"""
    snr_estimated = estimate_snr(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Signal-to-Noise Ratio Metric by Numpy: ", snr_estimated)


def test_segsnr():
    """Segmental Signal to Noise Ratio (SegSNR)"""
    snrseg_estimated = estimate_segsnr(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Segmental Signal to Noise Ratio Metric by Numpy: ", snrseg_estimated)


def test_llr():
    """Log-Likelihood Ratio (LLR)"""
    llr_estimated = estimate_llr(clean_speech, enhanced_speech, fs, window_size=128)
    print("-" * 80)
    print("Log-Likelihood Ratio(LLR) Metric by Numpy: ", llr_estimated)


def test_is():
    """Itakura-Saito distance measure (IS)"""
    is_estimated = estimate_is(clean_speech, enhanced_speech, fs, window_size=128)
    print("-" * 80)
    print("Itakura-Saito distance measure(IS) Metric by Numpy: ", is_estimated)


def test_cep():
    """Cepstrum distance measures (CEP)"""
    cep_estimated = estimate_cep(clean_speech, enhanced_speech, fs, window_size=128)
    print("-" * 80)
    print("Cepstrum distance measures Metric by Numpy: ", cep_estimated)


def test_fwssnr():
    """Frequency-weighted segmental SNR (fwSNRseg)"""
    fwssnr_estimated = estimate_fwssnr(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Frequency-weighted segmental SNR Metric by Numpy: ", fwssnr_estimated)


def test_wss():
    """Weighted spectral slope (WSS)"""
    wss_estimated = estimate_wss(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Weighted spectral slope Metric by Numpy: ", wss_estimated)


def test_sdr_several_source():
    """Signal Distortion Ratio (SDR) with musdb"""
    PATH_MUSDB18_7_STEMS = "./data/wav/musdb/"  # it need to download musdb18, refer https://github.com/sigsep/sigsep-mus-eval
    print("-" * 80)
    print(
        "Signal to Distortion Ratio (SDR) for several reference Metric by museval library: "
    )
    estimate_sdr_several_reference(data_path=PATH_MUSDB18_7_STEMS)


def test_sdr():
    """Signal Distortion Ratio (SDR)"""
    sdr_estimate = estimate_sdr_scratch(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Signal to Distortion Ratio Metric by Numpy: ", sdr_estimate)

    sdr_estimate = estimate_sdr(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Signal to Distortion Ratio Metric by museval library:", sdr_estimate)

    print("-" * 80)
    print("Signal to Distortion Ratio Metric by Torch: ", end="")
    sdr_estimate = torchmetrics.SignalDistortionRatio()
    sdr_estimate.update(preds=enhanced_speech_torch, target=clean_speech_torch)
    print(sdr_estimate.compute())


def test_si_snr():
    """Scale-Invariant Source-to-Noise Ratio(SI-SNR) same as SI-SDR"""
    si_snr_estimate = estimate_si_snr(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Scale-Invariant Source-to-Noise Ratio Metric by Numpy: ", si_snr_estimate)

    print("-" * 80)
    print("Scale-Invariant Source-to-Noise Ratio Metric by Torch: ", end="")
    si_snr_estimate = torchmetrics.ScaleInvariantSignalNoiseRatio()
    si_snr_estimate.update(preds=enhanced_speech_torch, target=clean_speech_torch)
    print(si_snr_estimate.compute())


def test_si_sdr():
    """Scale-Invariant Signal Distortion Ratio(SI-SDR)"""
    si_sdr_estimate = estimate_si_sdr(clean_speech, enhanced_speech, fs)
    print("-" * 80)
    print("Scale-Invariant Signal Distortion Ratio Metric by Numpy: ", si_sdr_estimate)

    print("-" * 80)
    print("Scale-Invariant Signal Distortion Ratio Metric by Torch: ", end="")
    si_sdr_estimate = torchmetrics.ScaleInvariantSignalDistortionRatio(zero_mean=True)
    si_sdr_estimate.update(preds=enhanced_speech_torch, target=clean_speech_torch)
    print(si_sdr_estimate.compute())


def test_stoi():
    """Short-Time Objective Intelligibility(STOI)"""
    stoi_estimate = estimate_stoi(clean_speech, enhanced_speech, fs, extended=False)
    print("-" * 80)
    print("Short term objective intelligibility Metric by Numpy: ", stoi_estimate)

    print("-" * 80)
    print("Short term objective intelligibility Metric by Torch: ", end="")
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    stoi_estimate = ShortTimeObjectiveIntelligibility(fs=fs)
    stoi_estimate.update(preds=enhanced_speech_torch, target=clean_speech_torch)
    print(stoi_estimate.compute())


def test_estoi():
    """Extended Short-Time Objective Intelligibility(STOI)"""
    extended_stoi_estimate = estimate_stoi(
        clean_speech, enhanced_speech, fs, extended=True
    )
    print("-" * 80)
    print(
        "Expanded Short term objective intelligibility Metric by Numpy: ",
        extended_stoi_estimate,
    )

    print("-" * 80)
    print("Expanded Short term objective intelligibility Metric by Torch: ", end="")
    from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

    extended_stoi_estimate = ShortTimeObjectiveIntelligibility(fs=fs, extended=True)
    extended_stoi_estimate.update(
        preds=enhanced_speech_torch, target=clean_speech_torch
    )
    print(extended_stoi_estimate.compute())


def test_pesq():
    """Perceptual Evaluation of Speech Quality (PESQ)"""
    from resampy import resample

    clean_speech_x2 = resample(clean_speech, fs, 16000)
    enhanced_speech_x2 = resample(enhanced_speech, fs, 16000)
    fs_x2 = 16000

    clean_speech_torch_x2 = torch.tensor(clean_speech_x2, dtype=torch.float64)
    enhanced_speech_torch_x2 = torch.tensor(enhanced_speech_x2, dtype=torch.float64)

    pesq_nb_estimate, mos_lqo_nb_estimate = estimate_pesq(
        clean_speech, enhanced_speech, fs, "nb"
    )  # narrowband
    pesq_wb_estimate, mos_lqo_wb_estimate = estimate_pesq(
        clean_speech_x2, enhanced_speech_x2, fs_x2, "wb"
    )  # wideband
    print("-" * 80)
    print(
        f"Perceptual Evaluation of Speech Quality Metric by Numpy: \n\
  1. MOS LQO\n\
narrow   band: {pesq_nb_estimate}\n\
wide     band: {pesq_wb_estimate}\n\
  2. PESQ MOS\n\
narrow   band: {mos_lqo_nb_estimate}\n\
wide     band: {mos_lqo_wb_estimate}\n"
    )

    del pesq_nb_estimate, pesq_wb_estimate

    print("-" * 80)
    print(
        "Perceptual Evaluation of Speech Quality Metric by Torch: ",
    )
    from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

    pesq_estimate = PerceptualEvaluationSpeechQuality(fs=fs, mode="nb")
    pesq_estimate.update(preds=enhanced_speech_torch, target=clean_speech_torch)
    print("narrow band:", pesq_estimate.compute())

    pesq_estimate = PerceptualEvaluationSpeechQuality(fs=fs_x2, mode="wb")
    pesq_estimate.update(preds=clean_speech_torch_x2, target=enhanced_speech_torch_x2)
    print("wide   band:", pesq_estimate.compute())

    del pesq_estimate


def test_composite():
    """Signal distortion(SIG), Background intrusiveness(BAK), The Overall Quality of Speech(OVRL)"""
    sig_estimate, bak_estimate, ovrl_estimate = estimate_composite(
        clean_speech, enhanced_speech, fs, mode="nb"
    )  # fs == 8000

    print("-" * 80)
    print(
        f"Metric by Numpy: \n\
SIG, signal distortion: {sig_estimate}\n\
BAK, background intrusiveness: {bak_estimate}\n\
OVA, mean opinion score: {ovrl_estimate}"
    )


if __name__ == "__main__":
    if len(PATH_CLEAN_SPEECH) == 0 or len(PATH_ENHANCED_SPEECH) == 0:
        raise ValueError(
            "Please specify the paths to the clean and enhanced speech files on the top line of this file."
        )

    print("Test Evaluation for Speech related Metrics")
    print("-" * 80)

    test_snr()
    test_segsnr()
    test_llr()
    test_is()
    test_cep()
    test_fwssnr()
    test_wss()
    test_sdr_several_source()
    test_sdr()
    test_si_sdr()
    test_si_snr()
    test_stoi()
    test_estoi()
    test_pesq()
    test_composite()

    print("-" * 80)
    print("Completed!")

Evaluation Metrics for Speech
=============================

This repository is for understanding several metrics evaluating the speech. Metrics 2-7 are based on in [Speech Enhancement book](https://www.routledge.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781138075573) written by Philipos C. Loizou and other metrics are based on each library release on python. The detail is described in the follows.

Evaluation Metrics
------------------
In all metrics, it needs 3 parameter, **reference signal, enhanced signal and sampling frequency**, which is not actually require in library but just for coherence. All exampls are inclueded in **main.py**.

$ Noise\ Signal = Enhanced\ Signal - Reference\ Signal $

1. Signal to Noise Ratio (SNR)

    $SNR_{dB} = log_{10} \dfrac{P_{Signal}}{P_{Noise}}$


2. Segmental Signal to Noise Ratio (SegSNR)
    > <span style="font-size:80%"> Below Metric 2-7 refers to pysepm library[2] and [Speech Enhancement book](https://www.routledge.com/Speech-Enhancement-Theory-and-Practice-Second-Edition/Loizou/p/book/9781138075573) and its [matlab code](https://github.com/IMLHF/Speech-Enhancement-Measures.git) written by Philipos C. Loizou.

3. Log-Likelihood Ratio (LLR)

    $d_{LLR}(\vec{a_c}, \vec{a_p}) = log{\Large(}\dfrac{\vec{a_p}R_c\vec{a_p}^T}{\vec{a_c}R_c\vec{a_c}^T}{\Large )}, [11]$

    $\vec{a_c}$ is the LPC vector of the reference speech signal frame and $\vec{a_c}$ is the LPC vector of the enhanced speech signal frame. $R_c$ is the autocorrelation matrix of the original speech signal. In [11, 13], only the smallest 95% of the fram LLR values were used to compute the average LLR value.

4. Itakura-Saito distance measure (IS)

    $d_{IS}(\vec{a_c}, \vec{a_p}) = \dfrac{\sigma_c^2}{\sigma_p^2}{\Large(} \dfrac{\vec{a_p}R_c\vec{a_p}^T}{\vec{a_c}R_c\vec{a_c}^T} {\Large )} + log{\Large (}\dfrac{\sigma_c^2}{\sigma_p^2}{\Large )} -1,\ [11]$

    $\sigma_c^2,\ \sigma_p^2$ are the LPC gains of the clean and enhanced signals. In [11], The IS values were limited in the range of [0, 100], which was necessary in order to minimize the number of outliers

5. Cepstrum distance measures (CEP)
    
    $c(m) = a_m + \displaystyle\sum_{k=1}^{m-1} \dfrac{k}{m}c(k)a_{m-k},\ 1 \leq n \leq p,\ [11]$
    
    $a_m$ is LPC coeeficients and the cepstrum distance,

    $d_{CEP}(\vec{c_c}, \vec{c_p}) = \dfrac{10}{log10} \sqrt{2\displaystyle\sum_{k=1}^p [c_c(k)-c_p(k)]^2}$

    $\vec{c_c}, \vec{c_p}$ are the cepstrum coeeficient vector, which means same as convolution with frame and LPC coefficients. In [11], this was limited in the range of [0, 10], which was necessary in order to minimize the number of outliers

6. Frequency-weighted segmental SNR (fwSNRseg)

    $fwSNRseg = \dfrac{10}{M} \times {\Large\displaystyle\sum_{m=1}^{M-1}} \dfrac{\displaystyle\sum_{j=1}^K W(j,m)log_{10}\frac{|X(j,m)|^2}{(|X(j,m)| - |\bar X(j,m)|)^2}}{ \displaystyle\sum_{j=1}^K  W(j, m)}, [11, 13]$

    In [11], it is used the specific window method and band. In the example, it also implies transfroming to octave band and window in [3].

7. Weighted spectral slope (WSS) 

    $d_{WSS} = \dfrac{1}{M} \displaystyle\sum_{m=1}^{M-1} \dfrac{\displaystyle\sum_{j=1}^K W(j,m)(S_c(j,m)-S_p(j,m))^2}{\displaystyle\sum_{j=1}^K W(j,m)^2}, [11]$

    W(j,m) are the weights, which computed in [2], K and M is defined depending on the number of bands. $S_c(j,m),\ S_p(j,m)$ are the splectral slopes for jth frequench and at frame m of clean and enhanced signals.


8. Signal Distortion Ratio (SDR)
    > <span style="font-size:80%"> Metric 8 example is from museval[5] and PytorchLightning Metrics[1].

    This concept used "Projector" between reference to reference in same frame, reference to reference in every frame, and reference to enhanced. It is used to compute the distortion ratio, which divied into $s_{target},\ e_{interference},\ e_{noise},\ e_{aritifical}$. The detail is in the paper [14]. 

9. Scale-Invariant Signal Distortion Ratio(SI-SDR)   
    > <span style="font-size:80%"> Metric 9 example is from PytorchLightning Metrics[1].
    
    $SI{\small-}SDR = 10log_{10} {\LARGE (} \dfrac{||\dfrac{\hat s^T s}{||s||^2}s||^2}{||\dfrac{\hat s^T s}{||s||^2}s-\hat s||^2}{\LARGE )}$

    It is also called as Scale-Invariant Source-to-Noise Ratio(SI-SNR)[1]. This replace SDR's 512-tap FIR to scaling. 


10. Short-Time Objective Intelligibility(STOI) [3,8,9,10] 
    > <span style="font-size:80%"> Metric 10 example is from pystoi library[3]. 
 
    *Cochlear Filter bank means Octave Filter band, $2^{1/3}$ below papers

    In short, the correlation between processed speech and clean speech based on Octave band (150-5KHz), considering SDR(Signal Distortion Ration) and size of STFT frame[8]. 

    In **Extended Version**, this method normalized the spectra and temporal, which exactly means rows and columns normalization and use average of inner product with clear speech and process one[10].

11. Perceptual Evaluation of Speech Quality (PESQ)
    > <span style="font-size:80%"> Metric 11 example is from python-pesq library[4].
    
    Its explanation was skipped since having complicated computation.

12. Signal distortion(SIG), Background intrusiveness(BAK), The Overall Quality of Speech(OVRL)
    
    This is the combination of metrics PESQ, WSS, LLR, and SEGSNR[2].

Test Audio Source
------------------
Clean and Noisy audio soruce from NOIZEUS.
NOIZEUS. The experiments were conducted using 13 sentences from the NOIZEUS noisy speech corpus. The corpus uses IEEE sentences downsampled from 25 kHz to 8 kHz.[6,11,12]


Environment
------------------
Test Envirnment is M1 Mac with conda-forge. 
This contains several packages freeze.yml for conda or requirements.txt for python, but it only needs below packages to be installed.

    - scipy
    - soundfile 
    - librosa
    - numpy
    - museval
    - pystoi
    - python-pesq
    - pytorch >= 0.1.8
    - torchmetrics

Reference
------------------
[1] [PytorchLighning/metrics](https://github.com/PyTorchLightning/metrics/blob/a971c6b456e40728b34494ff9186af20da46cb5b/torchmetrics/functional/audio/snr.py#L67)

[2] https://github.com/schmiph2/pysepm

[3] https://github.com/mpariente/pystoi

[4] https://github.com/ludlows/python-pesq

[5] https://github.com/sigsep/sigsep-mus-eval

[6] https://ecs.utdallas.edu/loizou/speech/noizeus/

[7] Y. Luo and N. Mesgarani, "TaSNet: Time-Domain Audio Separation Network for Real-Time, Single-Channel Speech
Separation," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp.
696-700, doi: 10.1109/ICASSP.2018.8462116.

[8] Taal, Cees H., et al. "A short-time objective intelligibility measure for time-frequency weighted noisy speech." *2010 IEEE international conference on acoustics, speech and signal processing*. IEEE, 2010.

[9] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', IEEE Transactions on Audio, Speech, and Language Processing, 2011.

[10] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated Noise Maskers', IEEE Transactions on Audio, Speech and Language Processing, 2016.

[11] Y. Hu and P. C. Loizou, "Subjective comparison and evaluation of speech enhancement algorithms,"
Speech commun., vol. 49, nos. 7–8, pp. 588–601, Jul. 2007.

[12] E. Rothauser, "IEEE recommended practice for speech quality measurements," IEEE Trans. Audio Electroacoustics, vol. 17, pp. 225–246, 1969.
    
[13] J. Hansen and B. Pellom, “An effective quality evaluation protocol for speech enhancement algorithms,” in Proc. Int. Conf. Spoken Lang. Process., 1998, vol. 7, pp. 2819–2822.

[14] Vincent, Emmanuel, Rémi Gribonval, and Cédric Févotte. "Performance measurement in blind audio source separation." IEEE transactions on audio, speech, and language processing 14.4 (2006): 1462-1469.

------------------
------------------

### TODO LIST

    1. Permutation invariant training (PIT)
    2. CSTI, Covariance-based STI procedure
        - Covariance-based STI procedure, [1,2]
        - STI procedure, [3]
    3. DAU, sophisticaed perceptual model, [4,5,6]
    4. Normalized Subband Envelope Correlation (NSEC), [7]

    Reference

    [1] R. L. Goldsworthy and J. E. Greenberg, “Analysis of speech- based speech transmission index methods with implications for nonlinear operations,” J. Acoust. Soc. Am., vol. 116, no. 6, pp. 3679–3689, 2004.

    [2] Koch. R., Geho ̈rgerechte Schallanalyse zur Vorhersage und Verbesserung der Sprachversta ̈ndlichkeit, Ph.D. thesis, Uni- versita ̈t Go ̈ttingen, 1992.

    [3] H. J. M. Steeneken and T. Houtgast, “A physical method for measuring speech-transmission quality,” J. Acoust. Soc. Am., vol. 67, no. 1, pp. 318–326, 1980.

    [4] T. Dau, D. Pu ̈schel, and A. Kohlrausch, “A quantitative model of the ”effective” signal processing in the auditory system. i. model structure,” J. Acoust. Soc. Am., vol. 99, no. 6, pp. 3615– 22, 1996.

    [5] C.Christiansen,“Speechintelligibilitypredictionoflinearand nonlinear processed speech in noise,” M.S. thesis, Technical University of Denmark, 2008.

    [6] C. H. Taal, R. C. Hendriks, R. Heusdens, J. Jensen, and U. Kjems, “An evaluation of objective quality measures for speech intelligibility prediction,” in Proc. Interspeech, 2009, pp. 1947–1950.

    [7] J. B. Boldt and D. P. W. Ellis, “A simple correlation-based model of intelligibility for nonlinear speech enhancement and separation,” in Proc. EUSIPCO, 2009, pp. 1849–1853.


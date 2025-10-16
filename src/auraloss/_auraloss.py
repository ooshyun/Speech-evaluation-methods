import torch
import auraloss

def estimate_mrsstftloss(clean_speech, enhanced_speech, fs):
    """Multi-Resolution STFT Loss
    
    Reference
    ---------
    [TODO]
    """
    params = {
        "sample_rate": 44100,
        "perceptual_weighting": True,
        "w_sc" : 0,
        "w_log_mag": 0,
        "w_lin_mag": 20
    }
    clean_speech = torch.tensor(clean_speech, dtype=torch.float32)
    enhanced_speech = torch.tensor(enhanced_speech, dtype=torch.float32)
    is_squeezed = False
    if len(clean_speech.shape) == 1:
        clean_speech = clean_speech.unsqueeze(0).unsqueeze(0)
        is_squeezed = True
    if len(enhanced_speech.shape) == 1:
        enhanced_speech = enhanced_speech.unsqueeze(0).unsqueeze(0)

    loss_function = auraloss.freq.MultiResolutionSTFTLoss(**params)
    mrstft_loss = loss_function(clean_speech, enhanced_speech).item()
    print("-" * 80)
    print("Multi-Resolution STFT Loss Metric by Numpy: ", mrstft_loss)
    return mrstft_loss
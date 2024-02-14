from typing import Dict, Tuple
import argparse
import datetime as dt
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch, torchaudio
from tqdm.auto import tqdm

# Import your own modules
from pflow.models.pflow_tts import pflowTTS
from pflow.text import sequence_to_text, text_to_sequence
from pflow.utils.model import denormalize, normalize
from pflow.utils.utils import get_user_data_dir, intersperse
from pflow.hifigan.config import v1
from pflow.hifigan.denoiser import Denoiser
from pflow.hifigan.env import AttrDict
from pflow.hifigan.models import Generator as HiFiGAN
from pflow.data.text_mel_datamodule import mel_spectrogram
from ukrainian_word_stress import Stressifier

# Constants
STRESS_SYMBOL = "ˈ"
PFLOW_CHECKPOINT = "checkpoints/elevenlabs/checkpoint_epoch=1249.ckpt"
HIFIGAN_CHECKPOINT = "HiFiGan/generator_v1"
WAV_PROMPT = "samples/gpt4_11labs_eni_000000000001.wav"
OUTPUT_FOLDER = "synth_output"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Stressifier
stressify = Stressifier(stress_symbol=STRESS_SYMBOL)


def load_model(checkpoint_path: str) -> pflowTTS:
    model = pflowTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device).eval()
    return model


def count_params(model: torch.nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters()):,}"


def load_vocoder(checkpoint_path: str) -> HiFiGAN:
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)['generator'])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan


def process_text(text: str) -> Dict[str, torch.Tensor]:
    """
    Processes input text for synthesis, including stressing and converting to phoneme sequence.

    Args:
    text (str): The input text to process.

    Returns:
    Dict[str, torch.Tensor]: A dictionary containing processed text information.
    """
    stressed_text = stressify(text)
    sequence = torch.tensor(intersperse(text_to_sequence(stressed_text, ['ukrainian_cleaners2']), 0), dtype=torch.long,
                            device=device)[None]
    sequence_lengths = torch.tensor([sequence.shape[-1]], dtype=torch.long, device=device)
    phonemes = sequence_to_text(sequence.squeeze(0).tolist())

    return {
        'x_orig': text,
        'x': sequence,
        'x_lengths': sequence_lengths,
        'x_phones': phonemes
    }

@torch.inference_mode()
def synthesise(text: str, prompt: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Synthesises speech from text using the pflowTTS model and a given prompt.

    Args:
    text (str): The input text to synthesise.
    prompt (torch.Tensor): The mel spectrogram prompt for conditioning the synthesis.

    Returns:
    Dict[str, torch.Tensor]: A dictionary containing the synthesis output and metadata.
    """
    text_processed = process_text(text)
    assert text_processed['x'].device == model.device, "Input tensor and model are on different devices"
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'],
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        length_scale=length_scale,
        prompt=prompt
    )
    output.update({
        'start_t': start_t,
        **text_processed
    })
    return output

@torch.inference_mode()
def to_waveform(mel: torch.Tensor, vocoder) -> torch.Tensor:
    """
    Converts mel spectrogram to waveform using the vocoder.

    Args:
    mel (torch.Tensor): The mel spectrogram to convert.
    vocoder: The vocoder model for conversion.

    Returns:
    torch.Tensor: The resulting audio waveform.
    """
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio


def save_to_folder(filename: str, output: Dict[str, torch.Tensor], folder: str) -> None:
    """
    Saves synthesised audio to a specified folder.

    Args:
    filename (str): The name of the file to save.
    output (Dict[str, torch.Tensor]): The output data containing the waveform to save.
    folder (str): The path to the folder where the audio will be saved.
    """
    path = Path(folder)
    path.mkdir(exist_ok=True, parents=True)
    waveform = output['waveform'].numpy()  # Assuming waveform is already moved to CPU
    sf.write(path / f'{filename}.wav', waveform, 22050, 'PCM_24')


model = load_model(PFLOW_CHECKPOINT)
vocoder = load_vocoder(HIFIGAN_CHECKPOINT)
denoiser = Denoiser(vocoder, mode='zeros')

## Number of ODE Solver steps
n_timesteps = 20

## Changes to the speaking rate
length_scale = 1.0

## Sampling temperature
temperature = 0.663


wav, sr = torchaudio.load(WAV_PROMPT)
prompt = mel_spectrogram(
            wav,
            1024,
            80,
            22050,
            256,
            1024,
            0,
            8000,
            center=False,
        )


prompt = prompt[:, :, :264]
prompt = normalize(prompt, model.mel_mean, model.mel_std)
prompt = prompt.to(device)

parser = argparse.ArgumentParser(description="Text-to-Speech Synthesis with pflowTTS and HiFiGAN")
parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
args = parser.parse_args()

# Now you can use args.text anywhere in your script to access the input text.
text = args.text

outputs, rtfs = [], []
rtfs_w = []

output = synthesise(text, prompt) #, torch.tensor([15], device=device, dtype=torch.long).unsqueeze(0))
output['waveform'] = to_waveform(output['mel'], vocoder)

# Compute Real Time Factor (RTF) with HiFi-GAN
t = (dt.datetime.now() - output['start_t']).total_seconds()
rtf_w = t * 22050 / (output['waveform'].shape[-1])

## Pretty print
print(f"{'*' * 53}")
print(f"Input text")
print(f"{'-' * 53}")
print(output['x_orig'])
print(f"{'*' * 53}")
print(f"Phonetised text")
print(f"{'-' * 53}")
print(output['x_phones'])
print(f"{'*' * 53}")
print(f"RTF:\t\t{output['rtf']:.6f}")
print(f"RTF Waveform:\t{rtf_w:.6f}")
rtfs.append(output['rtf'])
rtfs_w.append(rtf_w)

# Display the synthesised waveform
# ipd.display(ipd.Audio(output['waveform'], rate=22050))

## Save the generated waveform
save_to_folder(f"{int(time.time()*1000)}", output, OUTPUT_FOLDER)

print(f"Number of ODE steps: {n_timesteps}")
print(f"Mean RTF:\t\t\t\t{np.mean(rtfs):.6f} ± {np.std(rtfs):.6f}")
print(f"Mean RTF Waveform (incl. vocoder):\t{np.mean(rtfs_w):.6f} ± {np.std(rtfs_w):.6f}")
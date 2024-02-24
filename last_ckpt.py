import gradio as gr
import random
import sys
sys.path.append('..')

import datetime as dt
from pathlib import Path

import IPython.display as ipd
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm

from pflow.models.pflow_tts import pflowTTS
from pflow.text import sequence_to_text, text_to_sequence
from pflow.utils.model import denormalize
from pflow.utils.utils import get_user_data_dir, intersperse

from pflow.hifigan.config import v1
from pflow.hifigan.denoiser import Denoiser
from pflow.hifigan.env import AttrDict
from pflow.hifigan.models import Generator as HiFiGAN

import time
import random

import torchaudio
import glob
from pflow.utils.model import normalize
from pflow.data.text_mel_datamodule import mel_spectrogram
import matplotlib.pyplot as plt


device = "cuda:0"

get_user_data_dir()

PFLOW_CHECKPOINT = "/home/skravchenko/dev/opensource/pflowtts_pytorch_uk/logs/train/elevenlabs/runs/2024-02-20_00-51-04/checkpoints/last.ckpt" #fill in the path to the pflow checkpoint
# PFLOW_CHECKPOINT = "/home/skravchenko/dev/opensource/pflowtts_pytorch_uk/logs/train/elevenlabs/runs/2024-02-17_22-07-10/checkpoints/checkpoint_epoch=199.ckpt" #fill in the path to the pflow checkpoint

HIFIGAN_CHECKPOINT = "HiFiGan/generator_v1"
OUTPUT_FOLDER = "synth_output"

## Number of ODE Solver steps
n_timesteps = 20

## Changes to the speaking rate
length_scale = 1.0

## Sampling temperature
temperature = 0.663


model = pflowTTS.load_from_checkpoint(PFLOW_CHECKPOINT, map_location=device)
model = model.to(device)
model.eval()


h = AttrDict(v1)
hifigan = HiFiGAN(h).to(device)
hifigan.load_state_dict(torch.load(HIFIGAN_CHECKPOINT, map_location=device)['generator'])
hifigan_eval = hifigan.eval()
hifigan.remove_weight_norm()

denoiser = Denoiser(hifigan, mode='zeros')


wav_files = glob.glob("datasets/elevenlabs_dataset/wavs/*.wav") ## fill in the path to the LJSpeech-1.1 dataset
wav, sr = torchaudio.load(wav_files[1000])
prompt = mel_spectrogram(wav, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)
prompt = prompt[:, :, :264]
prompt = normalize(prompt, model.mel_mean, model.mel_std)
prompt = prompt.to(device)  # Move prompt to the GPU


@torch.inference_mode()
def process_text(text: str):
    x = torch.tensor(intersperse(text_to_sequence(text, ['ukrainian_cleaners']), 0), dtype=torch.long, device=device)[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    # ...
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }


@torch.inference_mode()
def synthesise(text, prompt, n_timesteps, temperature, length_scale, guidance_scale):
    text_processed = process_text(text)
    assert text_processed['x'].device == model.device, "Input tensor and model are on different devices"
    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed['x'],
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        length_scale=length_scale,
        prompt=prompt,
        guidance_scale=guidance_scale
    )
    # merge everything to one dict
    output.update({'start_t': start_t, **text_processed})
    return output


@torch.inference_mode()
def to_waveform(mel, vocoder):
    audio = vocoder(mel).clamp(-1, 1)
    audio = denoiser(audio.squeeze(0), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()


def save_to_folder(filename: str, output: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    # np.save(folder / f'{filename}', output['mel'].cpu().numpy())
    sf.write(folder / f'{filename}.wav', output['waveform'], 22050, 'PCM_24')


def clean_text(text):
    _pad = "_"
    _punctuation = '()-;:,.!?¡¿—…"«»“” '
    _letters_ipa = (
        "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ̯͡"
    )
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгґдеєжзиіїйклмнопрстуфхцчшщьюя"

    allowed_characters = set(_pad + _punctuation + _letters + _letters_ipa)

    return ''.join(filter(lambda char: char in allowed_characters, text))


def plot_spectrogram(mel, fs):
    """
    Plots a spectrogram.

    :param mel: Mel spectrogram to plot. Expected shape is (1, num_mels, time_steps).
    :param fs: Sampling frequency.
    :return: Path to the saved spectrogram image.
    """
    # Adjust for 3D input, assuming mel shape is (1, num_mels, time_steps)
    if mel.dim() == 3:
        mel = mel.squeeze(0)  # Remove the batch dimension if present

    plt.figure(figsize=(10, 4))
    plt.imshow(mel.cpu().numpy(), aspect='auto', origin='lower',
               interpolation='none')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    path = 'spectrogram.png'  # Save the plot to a file
    plt.savefig(path)
    plt.close()  # Close the plot to free memory
    return path


def gradio_synthesize(text, n_timesteps=20, temperature=0.663, length_scale=1.0, guidance_scale=0.0):
    global prompt
    global hifigan
    print(text)

    output = synthesise(text, prompt, n_timesteps, temperature, 1 / length_scale, guidance_scale)
    output['waveform'] = to_waveform(output['mel'], hifigan)
    waveform_np = output['waveform'].detach().cpu().numpy()  # Convert to NumPy array

    spectrogram_path = plot_spectrogram(output['mel'], 22050)  # Plot the spectrogram

    print(output['x_phones'])
    if waveform_np.ndim > 1:
        waveform_np = waveform_np.flatten()
    return output['x_phones'], (22050, waveform_np), spectrogram_path


# Create a Gradio interface
interface = gr.Interface(
    title="Text to Speech",
    description="### Text-to-Speech Synthesis Українською мовою.\nНормалізація тексту (цифри, дати, тощо в слова) та наголоси працюють лише частково, тому для наголосів варто використовувати символ [ˈ] після голосної літери у наголошеному складі, наприклад <власноˈруч>.\n\nДля синтезування аудіо використовується модель [pflowTTS] яка була натренована на синтетичному датасеті довжиною у 2 години і 20 хвилин, який був створений за допомогою elevenlabs.io.",
    allow_flagging="never",
    fn=gradio_synthesize,
    inputs=[
        gr.Textbox(label="Enter Text to Synthesize", value="Привіт"),
        gr.Slider(minimum=1, maximum=100, step=1, value=20, label="ODE Steps"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.66, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Speed of Speech"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=0.0, label="Guidance scale"),
    ],
    outputs=[
        gr.Textbox(label="Phones"),
        gr.Audio(interactive=False, label="Synthesized Audio"),
        gr.Image(label="Spectrogram")  # Add this line to display the spectrogram
    ],
    examples=[
        ["Так"],
        ["Ні"],
        ["Антропоцентризм"],
        ["Біотехнології"],
        ["Лінгвістика"],
        ["Інформатика"],
        ["Привіт"],
        ["Хто може підказати скільки спікерів потрібно для того, щоб піфлов почав клонувати голоси?"],
        ["А цей аудіо запис створено за допомогою натренованої піфлоˈв моделі текст ту спіч на синтетичному датасеˈті створеному за допомогою чат джіпіті чотири турбо та ілевен лабс."],
        ["Ти слідкуєш за останніми новинами у галузі електромобілів? Компанії як Тесла та Рівіан роблять величезний вплив на автомобільну індустрію."],
    ]
)

# Launch the app
interface.queue(api_open=False, max_size=15, default_concurrency_limit=3).launch(show_api=False, server_port=5856, server_name='0.0.0.0', share=False)

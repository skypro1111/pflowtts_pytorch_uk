import sys
sys.path.append('../..')

import argparse
import os
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

from pflow.cli import process_text, plot_spectrogram_to_numpy
from pflow.data.text_mel_datamodule import mel_spectrogram

def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided pflowTTS need some text to generate the waveforms."
    assert args.prompt, "Prompt wav must be provided"
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
    assert args.speaking_rate >= 0, "Speaking rate must be greater than 0"
    return args


def write_wavs(model, inputs, output_dir, external_vocoder=None):
    if external_vocoder is None:
        print("The provided model has the vocoder embedded in the graph.\nGenerating waveform directly")
        t0 = perf_counter()
        wavs, wav_lengths = model.run(None, inputs)
        infer_secs = perf_counter() - t0
        mel_infer_secs = vocoder_infer_secs = None
    else:
        print("Generating mel using pflowTTS")
        mel_t0 = perf_counter()
        mels, mel_lengths = model.run(None, inputs)
        mel_infer_secs = perf_counter() - mel_t0
        print("Generating waveform from mel using external vocoder")
        vocoder_inputs = {external_vocoder.get_inputs()[0].name: mels}
        vocoder_t0 = perf_counter()
        wavs = external_vocoder.run(None, vocoder_inputs)[0]
        vocoder_infer_secs = perf_counter() - vocoder_t0
        wavs = wavs.squeeze(1)
        wav_lengths = mel_lengths * 256
        infer_secs = mel_infer_secs + vocoder_infer_secs

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (wav, wav_length) in enumerate(zip(wavs, wav_lengths)):
        output_filename = output_dir.joinpath(f"output_{i + 1}.wav")
        audio = wav[:wav_length]
        print(f"Writing audio to {output_filename}")
        sf.write(output_filename, audio, 22050, "PCM_24")

    wav_secs = wav_lengths.sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    if mel_infer_secs is not None:
        mel_rtf = mel_infer_secs / wav_secs
        print(f"pflowTTS RTF: {mel_rtf}")
    if vocoder_infer_secs is not None:
        vocoder_rtf = vocoder_infer_secs / wav_secs
        print(f"Vocoder RTF: {vocoder_rtf}")
    print(f"Overall RTF: {rtf}")


def write_mels(model, inputs, output_dir):
    t0 = perf_counter()
    mels, mel_lengths = model.run(None, inputs)
    infer_secs = perf_counter() - t0

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, mel in enumerate(mels):
        output_stem = output_dir.joinpath(f"output_{i + 1}")
        plot_spectrogram_to_numpy(mel.squeeze(), output_stem.with_suffix(".png"))
        np.save(output_stem.with_suffix(".numpy"), mel)

    wav_secs = (mel_lengths * 256).sum() / 22050
    print(f"Inference seconds: {infer_secs}")
    print(f"Generated wav seconds: {wav_secs}")
    rtf = infer_secs / wav_secs
    print(f"RTF: {rtf}")


def main():
    parser = argparse.ArgumentParser(
        description="pflowTTS inference script"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="ONNX model to use",
    )
    parser.add_argument("--vocoder", type=str, default=None, help="Vocoder to use (defaults to None)")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    
    parser.add_argument("--prompt", type=str, default=None, help="Prompt wav file to use")
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.667,
        help="Variance of the x0 noise (default: 0.667)",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--gpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )

    args = parser.parse_args()
    args = validate_args(args)

    if args.gpu:
        providers = ["GPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    model = ort.InferenceSession(args.model, providers=providers)

    model_inputs = model.get_inputs()
    model_outputs = list(model.get_outputs())

    if args.text:
        text_lines = args.text.splitlines()
    else:
        with open(args.file, encoding="utf-8") as file:
            text_lines = file.read().splitlines()

    processed_lines = [process_text(0, line, "cpu") for line in text_lines]
    x = [line["x"].squeeze() for line in processed_lines]
    # Pad
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    x = x.detach().cpu().numpy()
    x_lengths = np.array([line["x_lengths"].item() for line in processed_lines], dtype=np.int64)
    # prompt wav
    wav_file = args.prompt
    prompt_wav, sr = torchaudio.load(wav_file)
    prompt = mel_spectrogram(
            prompt_wav,
            1024,
            80,
            22050,
            256,
            1024,
            0,
            8000,
            center=False,
        )
    
    inputs = {
        "x": x,
        "x_lengths": x_lengths,
        "prompt": np.array(prompt, dtype=np.float32),
        "scales": np.array([args.temperature, args.speaking_rate], dtype=np.float32),
    }
    
    has_vocoder_embedded = model_outputs[0].name == "wav"
    if has_vocoder_embedded:
        write_wavs(model, inputs, args.output_dir)
    elif args.vocoder:
        external_vocoder = ort.InferenceSession(args.vocoder, providers=providers)
        write_wavs(model, inputs, args.output_dir, external_vocoder=external_vocoder)
    else:
        warn = "[!] A vocoder is not embedded in the graph nor an external vocoder is provided. The mel output will be written as numpy arrays to `*.npy` files in the output directory"
        warnings.warn(warn, UserWarning)
        write_mels(model, inputs, args.output_dir)


if __name__ == "__main__":
    main()
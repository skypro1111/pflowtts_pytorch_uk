# pflowtts_pytorch_uk

## Overview

`pflowtts_pytorch_uk` is a modified version of the original [pflowtts_pytorch](https://github.com/p0p4k/pflowtts_pytorch) project, adapted specifically for training and inference in Ukrainian. This project aims to bring the advanced capabilities of neural network-based text-to-speech (TTS) to the Ukrainian language, leveraging the power of PyTorch for efficient and scalable deep learning. 

## Features

- **Ukrainian Language Support:** Customized to support the nuances of the Ukrainian language, including its phonetics and intonations, for natural and accurate speech synthesis.
- **Custom Training Dataset:** The training dataset was created using paid subscriptions to services like ChatGPT-4 for generating phonemically rich texts and ElevenLabs.io for audio generation. It contains a total of 2 hours and 20 minutes of audio, ensuring a high-quality and diverse dataset for training. Dataset available at [Hugging Face Datasets: skypro1111/elevenlabs_dataset](https://huggingface.co/datasets/skypro1111/elevenlabs_dataset).
- **Pretrained Models:** Includes pretrained models specifically tuned for Ukrainian speech, reducing the time needed to achieve high-quality TTS results. Despite the dataset's modest size of only 2 hours and 20 minutes of audio, the pretrained model checkpoints created based on this dataset demonstrate good performance, showcasing the effectiveness of the training data and techniques used. Checkpoints available at [Hugging Face Models: skypro1111/pyflowtts_uk_elevenlabs](https://huggingface.co/skypro1111/pyflowtts_uk_elevenlabs).

## Installation

To set up `pflowtts_pytorch_uk` on your system, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/skypro1111/pflowtts_pytorch_uk.git 
```
2. Install the required dependencies:
```bash
cd pflowtts_pytorch_uk
pip install -r requirements.txt
```
3. Build the Monotonic Alignment Search (MAS) module:
```bash
python setup.py build_ext --inplace
```
Instead of use Cython version of MAS, you can use JIT version (it's slower). For this you should uncomment 
```python
attn = (maximum_path_py(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach())
```
in pflow/models/pflow_tts.py forward method.

## Usage

### Training

To train a model with elevenlabs_dataset use the following command:
1. Generate normalisation statistics with the yaml file of dataset configuration:
```bash
cd pflowtts_pytorch_uk
export PROJECT_ROOT=.
python pflow/utils/generate_data_statistics.py -i elevenlabs.yaml
# Output:
#{'mel_mean': -4.698073863983154, 'mel_std': 1.8258213996887207}
```
Update these values in configs/data/elevenlabs.yaml under data_statistics key.
```bash
data_statistics:  # Computed for ljspeech dataset
  mel_mean: -4.698073863983154
  mel_std: 1.8258213996887207
```
2. Run the training script:
```bash
python pflow/train.py experiment=elevenlabs
```
 - for multi-gpu training, run:
```bash
python pflow/train.py experiment=elevenlabs trainer=ddp trainer.devices=[0,1,2]
```
* You can use CUDA_VISIBLE_DEVICES="1,2" before python command to specify the GPU to use.
* You can uncomment the Stressifier in pflow/text/cleaners.py to add stress to texts from the dataset during training, but this may lead to memory leaks. Alternatively, you can prepare the dataset by pre-stressing it, as in the stress.py example.

### Inference
To synthesize speech from text using a pretrained model, use the following command:
```bash
python synthesize.py --text "А цей датасет хоч і дуже малий, але дуже потужний."
```
* You should download the pretrained HiFiGan model checkpoint (and pflow checkpoint) from [Hugging Face Models: skypro1111/pyflowtts_uk_elevenlabs](https://huggingface.co/skypro1111/pyflowtts_uk_elevenlabs/blob/main/generator_v1) and place it in the `HiFiGan` directory.


## License

`pflowtts_pytorch_uk` is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

## Acknowledgments

This project is based on the [pflowtts_pytorch](https://github.com/p0p4k/pflowtts_pytorch) project by p0p4k. We extend our gratitude to the original authors for their groundbreaking work in the field of text-to-speech synthesis.

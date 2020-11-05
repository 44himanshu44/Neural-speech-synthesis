<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/as-ideas/TransformerTTS/master/docs/transformer_logo.png" width="400"/>
    <br>
</p>

<h2 align="center">
<p>A Text-to-Speech Transformer in TensorFlow 2</p>
</h2>

Implementation of a non-autoregressive Transformer based neural network for Text-to-Speech (TTS). <br>
This repo is based on the following papers:
- [Neural Speech Synthesis with Transformer Network](https://arxiv.org/abs/1809.08895)



#### Non-Autoregressive
Being non-autoregressive, this Transformer model is:
- Robust: No repeats and failed attention modes for challenging sentences.
- Fast: With no autoregression, predictions take a fraction of the time.
- Controllable: It is possible to control the speed of the generated utterance.

## 🔈 Sample output

Samples can be found [here](https://drive.google.com/drive/folders/1hhqgRnuYhU4LS6PBsyMhEgLVN6Taa8_R)

These sample spectrograms are converted using the pre-trained [melgan_Autoregressive_model_v2](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/TransformerTTS/ljspeech_melgan_autoregressive_transformer.zip) <br>


## 📖 Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Training](#training)
    - [Autoregressive](#train-autoregressive-model)
   
- [Prediction](#prediction)
- [Model Weights](#model_weights)

## Installation

Make sure you have:

* Python >= 3.6


Then install the rest with pip:
```
pip install -r requirements.txt
```

Read the individual scripts for more command line arguments.

## Dataset
You can directly use [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) to create the training dataset.
Or record your own voice samples

#### Configuration
* If training on LJSpeech, or if unsure, simply use one of 
    * ```config/melgan``` for models compatible with [MelGAN](https://github.com/seungwonpark/melgan) 
* **EDIT PATHS**: in `data_config.yaml` edit the paths to point at your dataset and log folders

#### Custom dataset
Prepare a dataset in the following format:
```
|- dataset_folder/
|   |- metadata.csv
|   |- wavs/
|       |- file1.wav
|       |- ...
```
where `metadata.csv` has the following format:
``` wav_file_name|transcription ```

## Training
Change the ```--config``` argument based on the configuration of your choice.
### Train Autoregressive Model
#### Create training dataset
```bash
python create_dataset.py --config config/melgan
```
#### Training
```bash
python train_autoregressive.py --config config/melgan
```

#### Training & Model configuration
- Training and model settings can be configured in `model_config.yaml`

#### Resume or restart training
- To resume training simply use the same configuration files AND `--session_name` flag, if any
- To restart training, delete the weights and/or the logs from the logs folder with the training flag `--reset_dir` (both) or `--reset_logs`, `--reset_weights`

#### Monitor training
We log some information that can be visualized with TensorBoard:
```bash
tensorboard --logdir /logs/directory/
```

![Tensorboard Demo](https://github.com/44himanshu44/Neural-speech-synthesis/blob/master/docs/tensorboard_monitor.gif)

## Prediction
Predict with either the Forward or Autoregressive model
```python
from utils.config_manager import ConfigManager
from utils.audio import Audio
import IPython.display as ipd

config_manager = ConfigManager(config_path='config/melgan', model_kind='autoregressive', session_name = None)
audio = Audio(config_manager.config)
conf_path = 'logdir/melgan/'
model = config_manager.load_model(conf_path +'autoregressive_weights/ckpt-18')
out = model.predict('Please, say something.')

# Convert spectrogram to wav (with griffin lim)
wav = audio.reconstruct_waveform(out['mel'].numpy().T)
ipd.display(ipd.Audio(wav, rate=config_manager.config['sampling_rate']))
```
## Model Weights
Fine tuned model on my voice :[melgan_autoregressive_model_ckpt_18](https://drive.google.com/file/d/1cxk6IwORIkX8jg1oLiLDB0TRw83NKZc0/view?usp=sharing)

## Refrence
See [TransformerTTS](https://github.com/as-ideas/TransformerTTS) for details.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import display

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.InteractiveSession(config=config)

train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
  directory='mini_speech_commands',
  batch_size=64,
  validation_split=0.2,
  seed=0,
  output_sequence_length=16000,
  subset='both'
)

labels = np.array(train_ds.class_names)
# print(train_ds.element_spec)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  
  return audio, labels

#The map function is used to apply the squeeze function to each element in the dataset. 
#The squeeze function is expected to take two arguments, an audio tensor and labels, and it returns the squeezed audio tensor and the unchanged labels.
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)#what is AUTOTUNE?
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

for i in range(3):
  label = labels[example_labels[i]]#?
  waveform = example_audio[i]#?
  spectrogram = get_spectrogram(waveform)#?

  print('Label:', label)
  print('Waveform shape:', waveform.shape)
  print('Spectrogram shape:', spectrogram.shape)
  print('Audio playback')
  display.display(display.Audio(waveform, rate=16000))
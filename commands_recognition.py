import tensorflow as tf
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.InteractiveSession(config=config)

DATA_PATH = 'mini_speech_commands'
classes = os.listdir(DATA_PATH)
audio_files = []
labels = []

for class_name in classes:
  
  for filename in os.listdir(DATA_PATH + f'/{class_name}'):
    audio_files.append(DATA_PATH + f'/{class_name}/{filename}')
    labels.append(class_name)

data = tf.data.Dataset.from_tensor_slices((audio_files, labels))
data = data.shuffle(len(data))

def loan_and_decode_audio(audio, label):
  audio_binary = tf.io.read_file(audio)
  waveform, _ = tf.audio.decode_wav(audio_binary, desired_samples=16000)
  
  return waveform, label

data = data.map(loan_and_decode_audio)#?

train_ds, val_ds = data.take(6400), data.skip(6400)
train_ds, val_ds = train_ds.batch(64), val_ds.batch(64)

def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  
  return audio, labels

train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

for example_audio, example_labels in train_ds.take(1):  
  print(example_audio.shape)
  print(example_labels.shape)

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram
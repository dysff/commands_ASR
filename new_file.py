import tensorflow as tf
import matplotlib.pyplot as plt
import os

DATA_PATH = 'mini_speech_commands'
INSTANCE = r'mini_speech_commands\down\0a9f9af7_nohash_0.wav'

def load_wav_16k_mono(filename):
  file_contents = tf.io.read_file(filename)
  wav, sr = tf.audio.decode_wav(file_contents, desired_samples=16000, desired_channels=1)#add resampling if needed
  wav = tf.squeeze(wav, axis=-1)
  
  return wav

#how to visualize waveform
# test_file = load_wav_16k_mono(INSTANCE)
# plt.plot(test_file)
# plt.show()

# file_contents = tf.io.read_file(INSTANCE)
# wav, sr = tf.audio.decode_wav(file_contents)
# print(wav)

classes = os.listdir(DATA_PATH)
audio_files = []
labels = []

for class_name in classes:
  
  for filename in os.listdir(DATA_PATH + f'/{class_name}'):
    audio_files.append(DATA_PATH + f'/{class_name}/{filename}')
    labels.append(class_name)

data = tf.data.Dataset.from_tensor_slices((audio_files, labels))#not shuffled
# print(data.as_numpy_iterator().next())

def preprocess(file_path, label):
  wav = load_wav_16k_mono(file_path)
  spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)#what is frame_length and frame_step
  spectrogram = tf.abs(spectrogram)
  spectrogram = tf.expand_dims(spectrogram, axis=2)#creates channel dimension

  return spectrogram, label

#spectrogram visualization
# filename, label = data.shuffle(8000).as_numpy_iterator().next()
# print(label)
# spectrogram, label = preprocess(filename, label)
# plt.figure(figsize=(30, 20))
# plt.imshow(tf.transpose(spectrogram)[0])
# plt.show()
import pyaudio
import numpy as np
import wave
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.InteractiveSession(config=config)

model = tf.keras.models.load_model('speech_recognition_model_v1.h5')

# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#   print(p.get_device_info_by_index(i))
  
# p.terminate()

def capture_audio():
    CHUNK = 1024  # Increase chunk size for longer duration
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_THRESHOLD = 1000
    SILENCE_CHUNKS_THRESHOLD = 2  # Adjust this as needed

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        input_device_index=1)

    print('Listening...')

    frames = []
    flag, frames_recorded, silence_counter, total_frames = 0, 0, 0, 0
    
    while True:
      data = stream.read(CHUNK)
      frames.append(data)
      audio_data = np.frombuffer(data, dtype=np.int16)# Convert audio data to numpy array
      silence_level = np.sum(audio_data ** 2) / len(audio_data)# Calculate silence_level of the audio frame
      # print(f'SILENCE_LEVEL: {silence_level}')
      
      if silence_level > RECORD_THRESHOLD:
        flag = 1
        frames_recorded += 1
        
      if flag:
        
        if silence_level < RECORD_THRESHOLD:
          silence_counter += 1
          
        else:
          silence_counter = 0
          
      if silence_counter > SILENCE_CHUNKS_THRESHOLD:
        flag = 0
        total_frames = frames_recorded
        
        print('-----------------RECORDED AUDIO INFO-----------------')
        print(f'TOTAL_FRAMES: {total_frames}')
        print(f'AMOUNT_OF_FRAMES: {len(frames)}')
        print('-----------------------------------------------------')
        
        frames_recorded = 0
        break

    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open('output_file.wav', 'wb') as wf:
      wf.setnchannels(CHANNELS)
      wf.setsampwidth(audio.get_sample_size(FORMAT))#what is get_sample_size
      wf.setframerate(RATE)
      wf.writeframes(b''.join(frames[-(total_frames + 12):]))
      wf.close()

def encode():
  DATA_PATH = r'mini_speech_commands'
  classes = os.listdir(DATA_PATH)
  label_encoder = LabelEncoder()
  label_encoder = label_encoder.fit(classes)
  
  return label_encoder

def load_wav_16k_mono(filename):
  file_contents = tf.io.read_file(filename)
  wav, sr = tf.audio.decode_wav(file_contents, desired_samples=16000, desired_channels=1)
  wav = tf.squeeze(wav, axis=-1)
  
  return wav

# def preprocess(filename):
#   wav = load_wav_16k_mono(filename)
#   spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)#what is frame_length and frame_step
#   spectrogram = tf.abs(spectrogram)
#   spectrogram = tf.expand_dims(spectrogram, axis=2)#creates channel dimension
  
#   return spectrogram

# spectrogramm = np.array(preprocess(r'mini_speech_commands\down\0a9f9af7_nohash_0.wav'))
# print(spectrogramm.shape)
# model = tf.keras.models.load_model('speech_recognition_model_v1.h5')
# prediction = model.predict(spectrogramm)
# print(prediction)

def preprocess(filename):
  wav = load_wav_16k_mono(filename)
  spectrogram = tf.signal.stft(wav, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = tf.expand_dims(spectrogram, axis=2)
  spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension
    
  return spectrogram

def speech_recognition():
  
  while True:
    capture_audio()
    spectrogram_tensor = preprocess('output_file.wav')
    prediction = model.predict(spectrogram_tensor)

    max_probability = np.max(prediction)
    if max_probability < 0.85:
        print('SPEECH IS NOT RECOGNIZED...')
        continue

    predicted_class_index = np.argmax(prediction)
    predicted_class_decoded = encode().inverse_transform([predicted_class_index])
    print('RECOGNIZED SPEECH: ', predicted_class_decoded, 'PROBABILITY: ', max_probability)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
from keras.utils import to_categorical

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
session = tf.compat.v1.InteractiveSession(config=config)

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
print(f'Classes: {classes}')
audio_files = []
labels = []

for class_name in classes:
  
  for filename in os.listdir(DATA_PATH + f'/{class_name}'):
    audio_files.append(DATA_PATH + f'/{class_name}/{filename}')
    labels.append(class_name)

# Encoding labels
def encoder(labels):
  label_encoder = LabelEncoder()
  label_encoder.fit(labels)
  labels_encoded = label_encoder.transform(labels)
  labels_encoded = to_categorical(labels_encoded)
  
  return labels_encoded

data = tf.data.Dataset.from_tensor_slices((audio_files, encoder(labels)))# now using encoded labels
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

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(8000)
data = data.batch(64)#64 samples at the time. How to choose this value properly?
data = data.prefetch(32)#what is it?

train, test = data.take(100), data.skip(100).take(25)
# print(train.element_spec)

#test one batch
samples, labels = train.as_numpy_iterator().next()
print(samples.shape)
print(labels)

#-----------------------------BUILDING THE MODEL----------------------------

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(124, 129, 1)))
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(124, 129, 1)))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(124, 129, 1)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(124, 129, 1)))

model.add(Flatten())

model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(classes), activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(), 
              loss='categorical_crossentropy', 
              metrics=[keras.metrics.Recall(), keras.metrics.Precision(), 'accuracy'])

stop_early = keras.callbacks.EarlyStopping(patience=15, monitor='val_accuracy')
history = model.fit(train, epochs=10 ** 10, validation_data=test, callbacks=[stop_early])

tf.keras.models.save_model(model, 'speech_recognition_model_v1.h5')

#learning curve visualization
learning_curve_data = history.history
precision = learning_curve_data['val_precision']
recall = learning_curve_data['val_recall']
accuracy = learning_curve_data['val_accuracy']

epochs = range(1, len(precision) + 1)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, precision, 'b', label='Validation Precision')
plt.plot(epochs, recall, 'r', label='Validation Recall')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Validation Precision and Recall')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'g', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
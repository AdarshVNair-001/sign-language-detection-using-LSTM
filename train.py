from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')
actions = np.load('actions.npy')

# Define the log directory for TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
#importing 3 lstm models
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))#passing 64 lstm units
model.add(LSTM(128, return_sequences=True, activation='relu'))#passing 128 lstm units
model.add(LSTM(64, return_sequences=False, activation='relu'))#passing 64 lstm units
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
model.summary()
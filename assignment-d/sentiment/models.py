from keras.models import Sequential
from keras import layers
import pandas as pd
import pickle

from preprocessing import PICKLE, PANDAS

## Read data
with open(PICKLE, 'rb') as f:
    cornell_dict = pickle.load(f)

with pd.HDFStore(PANDAS) as store:
    cornell_df = store.get('cornell_df') 

y_train = cornell_df[cornell_df['rater_id'] < 3]['y'].values
y_test  = cornell_df[cornell_df['rater_id'] == 3]['y'].values

midpoint = len(y_train)

x_train = cornell_dict['x_embed'][:midpoint]
x_test  = cornell_dict['x_embed'][midpoint:]


# 1D Convnet
conv_model = Sequential()
conv_model.add(layers.Conv1D(32, 7, activation='relu'))
conv_model.add(layers.MaxPool1D(5))
conv_model.add(layers.Conv1D(32, 7, activation='relu'))
conv_model.add(layers.GlobalAveragePooling1D())
conv_model.add(layers.Dense(1))

conv_model.compile(optimizer='rmsprop', loss='mse')
conv_model.fit(x_train, y_train, epochs=20, batch_size=128, shuffle=True)

score = conv_model.evaluate(x_test, y_test, batch_size=128)

print(score)
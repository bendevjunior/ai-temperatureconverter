import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
temperature_df = pd.read_csv('/workspaces/ai-temperatureconverter/Celsius-to-Fahrenheit.csv')
temperature_df.reset_index(drop=True, inplace=True)
# print(temperature_df)

# Visualizing the dataset
# sns.scatterplot(x=temperature_df['Celsius'], y=temperature_df['Fahrenheit'])

# Splitting the dataset into the Training set and Test set
x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']

# Building the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))
# model.summary()
# Compiling the model using optimizer and loss function 
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')

# Training the model
epochs_hist = model.fit(x_train, y_train, epochs=500)

# Visualizing the loss function
# epochs_hist.history.keys()
# plt.plot(epochs_hist.history['loss'])
# plt.title('Model Loss Progress During Training')
# plt.xlabel('Epoch')
# plt.ylabel('Training Loss')
# plt.legend(['Training Loss'])

#print(model.get_weights())

# Using the model to predict values
temp_c = 40
temp_f = model.predict([temp_c])
print('Temperature in Fahrenheit: ' + str(temp_f))
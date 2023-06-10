import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
temperature_df = pd.read_csv('/workspaces/ai-temperatureconverter/Celsius-to-Fahrenheit.csv')
temperature_df.reset_index(drop=True, inplace=True)
# print(temperature_df)

sns.scatterplot(x=temperature_df['Celsius'], y=temperature_df['Fahrenheit'])
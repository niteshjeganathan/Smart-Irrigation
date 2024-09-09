# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing Dataset
dataset = pd.read_csv('data.csv', skiprows=5)
X = dataset.iloc[:, [1, 2, 5, 6, 9, 11]].values

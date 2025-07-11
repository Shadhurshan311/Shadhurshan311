import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Step 1: Load Data
data = pd.read_csv("C:\\Users\\94772\\Desktop\\FYP\\Samples\\SampleOneFile.csv")
print("Available columns:", data.columns)

import seaborn as sns
print(sns.__version__)

data = data.drop_duplicates()
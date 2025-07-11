import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

import os
print(os.path.exists("C:\\Users\\94772\\Desktop\\FYP\\Samples\\Data_crack.csv"))

import pandas as pd
try:
    df = pd.read_csv(r"C:\Users\94772\Desktop\FYP\PythonApp\sampleONE.csv")
    print(df.head())
except Exception as e:
    print("Error:", e)


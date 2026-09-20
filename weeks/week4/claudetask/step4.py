import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("pk_dataset.csv")
numeric_column = ["dose", "A1", "A2", "C", "Time", "Volume", "CL", "Q"]
def clean_data(data):
    for column in numeric_column:
        data[column] = pd.to_numeric(data[column])
    return data

df = clean_data(df)
print(df)
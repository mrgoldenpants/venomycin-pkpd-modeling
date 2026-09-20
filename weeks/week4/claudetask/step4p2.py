import pandas as pd
import numpy as np

df = pd.read_csv("pk_dataset.csv")
numeric_column = ["dose", "A1", "A2", "C", "Time", "Volume", "CL", "Q"]
def clean_data(data):
    for column in numeric_column:
        data[column] = pd.to_numeric(data[column])
    return data


first_data = clean_data(df)
second_data = clean_data(df)

assert np.allclose(first_data, second_data)
import pandas as pd
import numpy as np


df = pd.read_csv("/Users/goldenpants/PycharmProjects/PythonProject/pk_dataset.csv")

sparse_df = df[df["Time"] % 20 == 0]

print(sparse_df)

print(np.random.seed(42))
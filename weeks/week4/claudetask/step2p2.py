import pandas as pd

df = pd.read_csv("pk_dataset.csv")
print(df)

assert pd.api.types.is_numeric_dtype(df["A1"])


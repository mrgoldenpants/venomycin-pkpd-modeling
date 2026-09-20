import pandas as pd

df = pd.read_csv("pk_dataset.csv")
df.describe()
print(df)

print(df.dtypes)



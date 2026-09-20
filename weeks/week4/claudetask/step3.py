import pandas as pd

df = pd.read_csv("pk_dataset.csv")
print(df)

observations = df.groupby("Subjects")["Time"].count
dose = df.groupby("Subjects")["dose"].apply(lambda x: (x>0).sum())
print(dose)
print(observations)

assert (dose > 0).all()
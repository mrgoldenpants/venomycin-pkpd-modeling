import pandas as pd

df = pd.read_csv("pk_dataset.csv")

#making sure everything is a number
numeric_column = ["dose", "A1", "A2", "C", "Time", "Volume", "CL", "Q"]
for column in numeric_column:
    df[column] = pd.to_numeric(df[column])

#removing negative concentrations
df = df[df["C"]>=0]

#remove duplicate timestamps
df = df.drop_duplicates(subset=["Subjects", "Time"])

#remove observations before the first dosage
first_dose_time = df[df["dose"] > 0]["Time"].min()
df = df[df["Time"] >= first_dose_time]

df.to_csv("cleaned_pk_dataset.csv", index=False)

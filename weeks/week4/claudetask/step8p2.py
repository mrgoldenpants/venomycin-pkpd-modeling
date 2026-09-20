import pandas as pd

df = pd.read_csv("pk_dataset.csv")

required_columns = [
    "Subjects",
    "dose",
    "A1",
    "A2",
    "C",
    "Time",
    "Volume",
    "CL",
    "Q"
]

df = df[required_columns].isna()
assert (~df[required_columns].isna()).all().all()
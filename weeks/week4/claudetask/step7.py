import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("pk_dataset.csv")
negative_concentration = df["C"] < 0
df["Negative concentration"] = negative_concentration

duplicate_time = df["Time"].duplicated()
df["Duplicate time"] = duplicate_time

first_dose_time = df[df["dose"] > 0]["Time"].min()
before_first_time = df["Time"] < first_dose_time
df["Before first time"] = before_first_time


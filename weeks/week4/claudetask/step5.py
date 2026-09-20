import pandas as pd
import numpy as np

df = pd.read_csv("pk_dataset.csv")

dosing_time = df[df["dose"]>0]["Time"].iloc[0]
observation_time = df["Time"]

assert (observation_time >= dosing_time).all
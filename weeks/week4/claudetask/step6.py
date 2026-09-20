import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("pk_dataset.csv")

for subject in df["Subjects"].unique():
    subject_data = df[df["Subjects"] == subject]
    plt.plot(subject_data["Time"], subject_data["C"])
    dose_times = subject_data[subject_data["dose"]>0]["Time"]
    for dose_time in dose_times:
        plt.axvline(x=dose_time, color="red")

plt.show()


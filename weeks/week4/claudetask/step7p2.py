import pandas as pd
df = pd.read_csv("pk_dataset.csv")
test_df = df.copy()
bad_row = {
    "Subjects": 1,
    "dose": 0,
    "A1": 50,
    "A2": 20,
    "C": -5,
    "Time": 50,
    "Volume": 10,
    "CL": 0.5,
    "Q": 0.2
}
test_df.loc[len(test_df)] = bad_row
negative_concentration = test_df["C"] < 0
test_df["Negative concentration"] = negative_concentration

assert test_df["Negative concentration"].iloc[-1] == True
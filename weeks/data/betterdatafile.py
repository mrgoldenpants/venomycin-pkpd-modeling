import pandas as pd

# Create a true PK time-series dataset using your variables
pk_data = {
    "Subjects": [1, 1, 1, 1, 1],
    "dose": [100, 0, 0, 0, 0],
    "A1": [100.0, 75.2, 56.5, 42.5, 31.9],
    "A2": [0.0, 18.3, 27.1, 30.5, 31.1],
    "C": [10.0, 7.52, 5.65, 4.25, 3.19],
    "Time": [0, 10, 20, 30, 40],
    "Volume": [10.0, 10.0, 10.0, 10.0, 10.0],
    "CL": [0.5, 0.5, 0.5, 0.5, 0.5],
    "Q": [0.2, 0.2, 0.2, 0.2, 0.2]
}

# Save as a numerical CSV
df_pk = pd.DataFrame(pk_data)
df_pk.to_csv("/Users/goldenpants/PycharmProjects/PythonProject/pk_dataset.csv", index=False)

print("Saved pk_dataset.csv!")
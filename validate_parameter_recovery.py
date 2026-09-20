"""Validate PK parameter recovery using known synthetic ground truth."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.fitting import fit_subject, predict_concentration


TRUE_PARAMETERS = {
    "CL": 0.5,
    "Q": 0.2,
    "V1": 10.0,
    "V2": 15.0,
}

OBSERVATION_TIME = np.array(
    [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]
)

INITIAL_GUESSES = {
    "Start 1": np.array([1.0, 0.5, 8.0, 15.0]),
    "Start 2": np.array([0.25, 0.10, 12.0, 20.0]),
    "Start 3": np.array([2.0, 2.0, 10.0, 15.0]),
    "Start 4": np.array([4.0, 4.0, 4.0, 4.0]),
}

RESULTS_DIRECTORY = Path("results")
RESULTS_DIRECTORY.mkdir(exist_ok=True)


def main() -> None:
    parameter_names = ["CL", "Q", "V1", "V2"]
    true_theta = np.array(
        [TRUE_PARAMETERS[name] for name in parameter_names]
    )

    # Generate noise-free observations from the known parameters.
    synthetic_concentration = predict_concentration(
        true_theta,
        OBSERVATION_TIME,
        dose=100.0,
    )

    synthetic_data = pd.DataFrame(
        {
            "Time": OBSERVATION_TIME,
            "Concentration": synthetic_concentration,
        }
    )

    synthetic_data.to_csv(
        RESULTS_DIRECTORY / "synthetic_validation_data.csv",
        index=False,
    )

    fit_records = []
    successful_fits = []

    for start_name, initial_guess in INITIAL_GUESSES.items():
        result = fit_subject(
            observed_time=OBSERVATION_TIME,
            observed_concentration=synthetic_concentration,
            initial_guess=initial_guess,
            dose=100.0,
        )

        predicted = predict_concentration(
            result.x,
            OBSERVATION_TIME,
            dose=100.0,
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (synthetic_concentration - predicted) ** 2
                )
            )
        )

        fit_records.append(
            {
                "Starting point": start_name,
                "Initial CL": initial_guess[0],
                "Initial Q": initial_guess[1],
                "Initial V1": initial_guess[2],
                "Initial V2": initial_guess[3],
                "Fitted CL": result.x[0],
                "Fitted Q": result.x[1],
                "Fitted V1": result.x[2],
                "Fitted V2": result.x[3],
                "RMSE": rmse,
                "Success": result.success,
            }
        )

        successful_fits.append((rmse, result))

    multistart_table = pd.DataFrame(fit_records).sort_values("RMSE")

    multistart_table.to_csv(
        RESULTS_DIRECTORY / "parameter_recovery_multistart.csv",
        index=False,
    )

    # Use the fit with the lowest RMSE rather than choosing a start manually.
    best_rmse, best_result = min(
        successful_fits,
        key=lambda item: item[0],
    )

    recovery_records = []

    for name, true_value, fitted_value in zip(
        parameter_names,
        true_theta,
        best_result.x,
    ):
        percent_error = (
            abs(fitted_value - true_value) / abs(true_value)
        ) * 100.0

        recovery_records.append(
            {
                "Parameter": name,
                "True": true_value,
                "Fitted": fitted_value,
                "Percent Error": percent_error,
            }
        )

    recovery_table = pd.DataFrame(recovery_records)

    recovery_table.to_csv(
        RESULTS_DIRECTORY / "parameter_recovery.csv",
        index=False,
    )

    print("\nSYNTHETIC VALIDATION DATA")
    print(synthetic_data.to_string(index=False))

    print("\nMULTISTART FIT RESULTS")
    print(multistart_table.to_string(index=False))

    print("\nPARAMETER-RECOVERY TABLE")
    print(
        recovery_table.to_string(
            index=False,
            formatters={
                "True": "{:.4f}".format,
                "Fitted": "{:.4f}".format,
                "Percent Error": "{:.2f}%".format,
            },
        )
    )

    print(f"\nBest concentration RMSE: {best_rmse:.6f} mg/L")

    largest_error_row = recovery_table.loc[
        recovery_table["Percent Error"].idxmax()
    ]

    print("\nPOSSIBLE README CAPTION")
    print(
        "Using noise-free synthetic observations with known ground truth, "
        f"the best multistart fit achieved an RMSE of "
        f"{best_rmse:.6f} mg/L; the largest parameter error was for "
        f"{largest_error_row['Parameter']} "
        f"({largest_error_row['Percent Error']:.2f}%)."
    )


if __name__ == "__main__":
    main()
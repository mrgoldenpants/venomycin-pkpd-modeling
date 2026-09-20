"""Check convergence of Sobol S1 and ST sensitivity indices."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from src.sensitivity import DEFAULT_PROBLEM, evaluate_parameter_samples


N_VALUES = [128, 256, 512, 1024, 2048]
SEED = 42
OUTPUT_NAME = "auc_0_24"

RESULTS_DIRECTORY = Path("results")
RESULTS_DIRECTORY.mkdir(exist_ok=True)


def main() -> None:
    records = []

    for n_base in N_VALUES:
        start_time = perf_counter()

        print(f"\nRunning Sobol analysis with N = {n_base}...")

        parameter_samples = sobol_sample.sample(
            DEFAULT_PROBLEM,
            n_base,
            calc_second_order=False,
            seed=SEED,
        )

        evaluated_outputs = evaluate_parameter_samples(
            parameter_samples,
            dose=100.0,
        )

        sobol_result = sobol_analyze.analyze(
            DEFAULT_PROBLEM,
            evaluated_outputs[OUTPUT_NAME],
            calc_second_order=False,
            print_to_console=False,
        )

        elapsed_time = perf_counter() - start_time

        for parameter, s1, st, s1_conf, st_conf in zip(
            DEFAULT_PROBLEM["names"],
            sobol_result["S1"],
            sobol_result["ST"],
            sobol_result["S1_conf"],
            sobol_result["ST_conf"],
        ):
            records.append(
                {
                    "N": n_base,
                    "Total evaluations": len(parameter_samples),
                    "Parameter": parameter,
                    "S1": float(s1),
                    "S1 confidence": float(s1_conf),
                    "ST": float(st),
                    "ST confidence": float(st_conf),
                    "Runtime seconds": elapsed_time,
                }
            )

        print(f"Completed N = {n_base} in {elapsed_time:.1f} seconds.")

    convergence_table = pd.DataFrame(records)

    convergence_table.to_csv(
        RESULTS_DIRECTORY / "sobol_convergence.csv",
        index=False,
    )

    print("\nFULL SOBOL CONVERGENCE TABLE")
    print(
        convergence_table[
            ["N", "Total evaluations", "Parameter", "S1", "ST"]
        ].to_string(
            index=False,
            formatters={
                "S1": "{:.6f}".format,
                "ST": "{:.6f}".format,
            },
        )
    )

    final_results = convergence_table[
        convergence_table["N"] == N_VALUES[-1]
    ]

    print(f"\nFINAL SOBOL RESULTS: N = {N_VALUES[-1]}")
    print(
        final_results[
            [
                "Parameter",
                "S1",
                "S1 confidence",
                "ST",
                "ST confidence",
            ]
        ].to_string(
            index=False,
            formatters={
                "S1": "{:.6f}".format,
                "S1 confidence": "{:.6f}".format,
                "ST": "{:.6f}".format,
                "ST confidence": "{:.6f}".format,
            },
        )
    )

    maximum_st = final_results["ST"].max()

    print(f"\nMaximum final ST: {maximum_st:.6f}")

    if maximum_st <= 1.0:
        print("All final total-order indices are at or below 1.")
    else:
        print(
            "At least one final ST remains above 1; inspect convergence "
            "and confidence intervals before interpreting it."
        )

    print(
        "\nResults saved to:",
        RESULTS_DIRECTORY / "sobol_convergence.csv",
    )


if __name__ == "__main__":
    main()
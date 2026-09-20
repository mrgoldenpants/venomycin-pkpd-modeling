"""Generate key figures for the PK/PD project README."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

from src.fitting import fit_subject, predict_concentration
from src.models import PKPDParams
from src.population import generate_virtual_patients, simulate_population
from src.sensitivity import DEFAULT_PROBLEM, evaluate_parameter_samples



# Create the output folder automatically.
FIGURES_DIRECTORY = Path("figures")
FIGURES_DIRECTORY.mkdir(exist_ok=True)


def make_fitted_vs_observed_figure() -> str:
    """Create a fitted-versus-observed concentration figure."""

    observed_time = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    observed_concentration = np.array([10.0, 7.52, 5.65, 4.25, 3.19])

    fit_result = fit_subject(
        observed_time=observed_time,
        observed_concentration=observed_concentration,
        dose=100.0,
    )

    plot_time = np.linspace(0.0, 48.0, 300)
    fitted_concentration = predict_concentration(
        fit_result.x,
        plot_time,
        dose=100.0,
    )

    predicted_at_observations = predict_concentration(
        fit_result.x,
        observed_time,
        dose=100.0,
    )

    rmse = np.sqrt(
        np.mean(
            (observed_concentration - predicted_at_observations) ** 2
        )
    )

    plt.figure(figsize=(8, 5))

    plt.scatter(
        observed_time,
        observed_concentration,
        color="black",
        s=55,
        label="Observed concentrations",
        zorder=3,
    )

    plt.plot(
        plot_time,
        fitted_concentration,
        color="tab:blue",
        linewidth=2.5,
        label="Fitted two-compartment model",
    )

    plt.xlabel("Time (hours)")
    plt.ylabel("Central concentration (mg/L)")
    plt.title("Fitted vs. Observed Concentrations")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIRECTORY / "fitted_vs_observed.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return (
        "The fitted two-compartment model followed the illustrative observed "
        f"concentrations with an RMSE of {rmse:.3f} mg/L."
    )


def make_sobol_sensitivity_figure() -> str:
    """Create a converged Sobol sensitivity figure for AUC0-24."""

    n_base = 2048
    seed = 42

    parameter_samples = sobol_sample.sample(
        DEFAULT_PROBLEM,
        n_base,
        calc_second_order=False,
        seed=seed,
    )

    evaluated_outputs = evaluate_parameter_samples(
        parameter_samples,
        dose=100.0,
    )

    sobol_result = sobol_analyze.analyze(
        DEFAULT_PROBLEM,
        evaluated_outputs["auc_0_24"],
        calc_second_order=False,
        print_to_console=False,
    )

    parameter_names = DEFAULT_PROBLEM["names"]
    first_order_indices = np.asarray(sobol_result["S1"])
    total_order_indices = np.asarray(sobol_result["ST"])

    x_positions = np.arange(len(parameter_names))
    bar_width = 0.36

    plt.figure(figsize=(8, 5))

    plt.bar(
        x_positions - bar_width / 2,
        first_order_indices,
        width=bar_width,
        color="tab:blue",
        label="First-order index (S1)",
    )

    plt.bar(
        x_positions + bar_width / 2,
        total_order_indices,
        width=bar_width,
        color="tab:purple",
        label="Total-order index (ST)",
    )

    plt.xticks(x_positions, parameter_names)
    plt.xlabel("PK parameter")
    plt.ylabel("Sobol sensitivity index")
    plt.title(
        f"Global Sensitivity of AUC$_{{0-24}}$ "
        f"(Sobol N = {n_base:,})"
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIRECTORY / "sobol_sensitivity.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    dominant_index = int(np.nanargmax(total_order_indices))
    dominant_parameter = parameter_names[dominant_index]
    dominant_value = total_order_indices[dominant_index]

    return (
        f"At a Sobol base sample size of {n_base:,}, "
        f"{dominant_parameter} was the most influential parameter for "
        f"AUC0–24, with a total-order index of {dominant_value:.3f}."
    )


def make_population_histogram() -> str:
    """Create a virtual-population AUC24 target-attainment histogram."""

    params = PKPDParams()

    patients = generate_virtual_patients(
        n=1000,
        seed=42,
    )

    population_output = simulate_population(
        patients=patients,
        dose=1000.0,
        tau=12.0,
        include_profiles=False,
    )

    results = population_output["results"]
    auc24_values = results["AUC24"].to_numpy()
    target_attainment = results["target_attained"].mean()

    plt.figure(figsize=(8, 5))

    plt.hist(
        auc24_values,
        bins=30,
        color="tab:green",
        edgecolor="white",
        alpha=0.85,
    )

    plt.axvspan(
        params.AUC24_lower,
        params.AUC24_upper,
        color="gold",
        alpha=0.25,
        label="Illustrative AUC24 target",
    )

    plt.axvline(
        params.AUC24_lower,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
    )

    plt.axvline(
        params.AUC24_upper,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
    )

    plt.xlabel("AUC24 (mg·h/L)")
    plt.ylabel("Number of virtual patients")
    plt.title(
        "Virtual-Patient AUC24 Distribution\n"
        f"1000 mg q12h; PTA = {target_attainment:.1%}"
    )
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIRECTORY / "population_target_attainment.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return (
        f"The illustrative 1000 mg q12h regimen placed "
        f"{target_attainment:.1%} of virtual patients within the "
        f"{params.AUC24_lower:.0f}–{params.AUC24_upper:.0f} mg·h/L "
        "AUC24 target range."
    )


def main() -> None:
    """Generate all README figures and print their captions."""

    fitted_caption = make_fitted_vs_observed_figure()
    sobol_caption = make_sobol_sensitivity_figure()
    population_caption = make_population_histogram()

    print("\nFigures created successfully:")
    print("1. figures/fitted_vs_observed.png")
    print("2. figures/sobol_sensitivity.png")
    print("3. figures/population_target_attainment.png")

    print("\nREADME captions:")
    print(f"1. {fitted_caption}")
    print(f"2. {sobol_caption}")
    print(f"3. {population_caption}")


if __name__ == "__main__":
    main()
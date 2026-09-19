"""Local and Sobol global sensitivity-analysis functions."""
"""
sensitivity.py investigates how uncertainty or variation in PK parameters
affects model outputs. It includes one-at-a-time local perturbations, Sobol
sampling and analysis, parameter ranking, and interaction contributions.
"""
import numpy as np

from .models import PKPDParams
from .simulation import run_pipeline


DEFAULT_PROBLEM = {
    "num_vars": 4,
    "names": ["CL", "Q", "V1", "V2"],
    "bounds": [
        [0.5, 25.0],
        [2.0, 30.0],
        [3.0, 8.0],
        [15.0, 150.0],
    ],
}


def perturb_parameter(params, parameter, fractional_change=0.10):
    """Return a parameter set with exactly one PK parameter changed."""
    allowed = {"CL", "Q", "V1", "V2"}
    if parameter not in allowed:
        raise ValueError(f"parameter must be one of {sorted(allowed)}")
    return params._replace(
        **{parameter: getattr(params, parameter) * (1.0 + fractional_change)}
    )


def calculate_local_sensitivity(
    dose=100.0,
    params=PKPDParams(),
    fractional_change=0.10,
    output="auc_0_48",
):
    """Return percent output changes from one-at-a-time perturbations."""
    baseline = run_pipeline(dose, params)[output]
    if baseline == 0:
        raise ValueError("Baseline output cannot be zero.")
    changes = {}
    for parameter in ("CL", "Q", "V1", "V2"):
        changed_params = perturb_parameter(params, parameter, fractional_change)
        changed_output = run_pipeline(dose, changed_params)[output]
        changes[parameter] = ((changed_output - baseline) / baseline) * 100.0
    return changes


def rank_local_sensitivity(changes):
    """Rank local percent changes by absolute magnitude."""
    return sorted(changes.items(), key=lambda item: abs(item[1]), reverse=True)


def generate_sobol_samples(problem=DEFAULT_PROBLEM, n_base=1024, seed=None):
    """Generate Sobol samples using SALib."""
    try:
        from SALib.sample import sobol
    except ImportError as error:
        raise ImportError("Install SALib to generate Sobol samples.") from error
    return sobol.sample(
        problem,
        n_base,
        calc_second_order=True,
        seed=seed,
    )


def evaluate_parameter_samples(
    parameter_samples,
    dose=100.0,
    base_params=PKPDParams(),
):
    """Evaluate AUC, AUC24, Cmax, and trough on identical parameter samples."""
    samples = np.asarray(parameter_samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError("parameter_samples must have shape (n, 4).")
    outputs = {
        "auc_0_48": [],
        "auc_0_24": [],
        "cmax": [],
        "trough": [],
    }
    for CL, Q, V1, V2 in samples:
        params = base_params._replace(CL=CL, Q=Q, V1=V1, V2=V2)
        result = run_pipeline(dose, params)
        for name in outputs:
            outputs[name].append(result[name])
    return {name: np.asarray(values) for name, values in outputs.items()}


def run_sobol_analysis(problem, output_values):
    """Calculate first- and total-order Sobol sensitivity indices."""
    try:
        from SALib.analyze import sobol
    except ImportError as error:
        raise ImportError("Install SALib to run Sobol analysis.") from error
    return sobol.analyze(
        problem,
        np.asarray(output_values, dtype=float),
        calc_second_order=True,
        print_to_console=False,
    )


def analyze_multiple_outputs(problem, evaluated_outputs):
    """Run Sobol analysis separately for every evaluated output."""
    lengths = {len(values) for values in evaluated_outputs.values()}
    if len(lengths) != 1:
        raise ValueError("Every output must use the same number of samples.")
    return {
        name: run_sobol_analysis(problem, values)
        for name, values in evaluated_outputs.items()
    }


def calculate_interaction_contribution(sobol_result):
    """Return ST minus S1 for each parameter."""
    contribution = np.asarray(sobol_result["ST"]) - np.asarray(
        sobol_result["S1"]
    )
    return np.maximum(contribution, 0.0)


def rank_sensitivity_indices(problem, sobol_result, index="ST"):
    """Rank named parameters by a selected Sobol index."""
    if index not in {"S1", "ST"}:
        raise ValueError("index must be 'S1' or 'ST'.")
    return sorted(
        zip(problem["names"], sobol_result[index]),
        key=lambda item: item[1],
        reverse=True,
    )

"""Virtual-patient generation and target-attainment functions."""
"""
population.py models between-patient variability using virtual patients. It
generates positive PK-parameter samples, simulates a shared population under
one or more regimens, and calculates probability of target attainment.
"""
import numpy as np
import pandas as pd

from .metrics import calculate_auc_mic
from .models import PKPDParams
from .optimization import regimen_auc24
from .simulation import run_pipeline


def lognormal_parameters(mean, coefficient_of_variation):
    """Convert arithmetic mean and CV to log-space mu and sigma."""
    if mean <= 0 or coefficient_of_variation < 0:
        raise ValueError("mean must be positive and CV cannot be negative.")
    sigma = np.sqrt(np.log(1.0 + coefficient_of_variation**2))
    mu = np.log(mean) - sigma**2 / 2.0
    return float(mu), float(sigma)


def generate_virtual_patients(
    n=1000,
    seed=42,
    means=None,
    coefficients_of_variation=None,
):
    """Generate positive lognormally distributed CL, Q, V1, and V2 values."""
    if n <= 0:
        raise ValueError("n must be greater than zero.")
    means = means or {"CL": 5.0, "Q": 2.0, "V1": 10.0, "V2": 20.0}
    coefficients_of_variation = coefficients_of_variation or {
        "CL": 0.30,
        "Q": 0.30,
        "V1": 0.25,
        "V2": 0.25,
    }
    rng = np.random.default_rng(seed)
    patients = {"patient_id": np.arange(n)}
    for parameter in ("CL", "Q", "V1", "V2"):
        mu, sigma = lognormal_parameters(
            means[parameter], coefficients_of_variation[parameter]
        )
        patients[parameter] = rng.lognormal(mu, sigma, size=n)
    return pd.DataFrame(patients)


def calculate_target_attainment(
    values,
    lower=400.0,
    upper=600.0,
) -> float:
    """Return the fraction of exposure values inside the target interval."""
    values = np.asarray(values, dtype=float)
    if values.size == 0 or lower > upper:
        raise ValueError("Provide exposure values and valid target bounds.")
    return float(np.mean((values >= lower) & (values <= upper)))


def simulate_population(
    patients,
    dose,
    tau,
    base_params=PKPDParams(),
    include_profiles=True,
):
    """Run one shared virtual population under a dosing regimen."""
    required = {"patient_id", "CL", "Q", "V1", "V2"}
    if not required.issubset(patients.columns):
        raise ValueError(f"patients must contain columns {sorted(required)}")

    records = []
    profiles = []
    for patient in patients.itertuples(index=False):
        params = base_params._replace(
            CL=patient.CL,
            Q=patient.Q,
            V1=patient.V1,
            V2=patient.V2,
        )
        auc24 = regimen_auc24(dose, tau, params.CL)
        auc_mic = calculate_auc_mic(auc24, params.MIC)
        records.append(
            {
                "patient_id": patient.patient_id,
                "CL": patient.CL,
                "Q": patient.Q,
                "V1": patient.V1,
                "V2": patient.V2,
                "dose": dose,
                "tau": tau,
                "AUC24": auc24,
                "AUC_MIC": auc_mic,
                "target_attained": (
                    params.AUC24_lower <= auc24 <= params.AUC24_upper
                ),
            }
        )
        if include_profiles:
            profiles.append(run_pipeline(dose, params)["concentration"])

    return {
        "patients": patients.copy(),
        "results": pd.DataFrame(records),
        "profiles": np.asarray(profiles) if include_profiles else None,
    }


def compare_regimens(
    patients,
    regimens,
    base_params=PKPDParams(),
    include_profiles=False,
):
    """Compare regimens using the exact same virtual-patient table."""
    comparisons = {}
    original_ids = patients["patient_id"].to_numpy(copy=True)
    for dose, tau in regimens:
        simulation = simulate_population(
            patients,
            dose,
            tau,
            base_params,
            include_profiles=include_profiles,
        )
        if not np.array_equal(
            simulation["patients"]["patient_id"].to_numpy(), original_ids
        ):
            raise RuntimeError("Regimens did not use identical virtual patients.")
        results = simulation["results"]
        simulation["pta"] = float(results["target_attained"].mean())
        comparisons[(dose, tau)] = simulation
    return comparisons

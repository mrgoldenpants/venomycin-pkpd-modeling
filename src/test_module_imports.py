"""Smoke tests proving reusable project functions work without notebooks."""

import importlib.util

import numpy as np
import pandas as pd
import pytest


def test_every_module_imports_without_a_notebook():
    """Every moved module is importable as a normal Python module."""
    import src.data_processing
    import src.diagnostics
    import src.fitting
    import src.metrics
    import src.models
    import src.optimization
    import src.population
    import src.sensitivity
    import src.simulation


def test_population_simulation_is_reproducible_with_same_seed():
    import pandas as pd

    from src.population import generate_virtual_patients, simulate_population

    patients_1 = generate_virtual_patients(n=100, seed=42)
    patients_2 = generate_virtual_patients(n=100, seed=42)

    simulation_1 = simulate_population(
        patients_1,
        dose=1000.0,
        tau=12.0,
        include_profiles=False,
    )
    simulation_2 = simulate_population(
        patients_2,
        dose=1000.0,
        tau=12.0,
        include_profiles=False,
    )

    pd.testing.assert_frame_equal(
        simulation_1["patients"],
        simulation_2["patients"],
    )
    pd.testing.assert_frame_equal(
        simulation_1["results"],
        simulation_2["results"],
    )


def test_models_metrics_simulation_and_diagnostics_are_callable():
    from src.diagnostics import (
        absolute_error,
        calculate_aic,
        calculate_bic,
        calculate_mae,
        calculate_mean_residual,
        calculate_r_squared,
        calculate_rmse,
        calculate_sse,
        goodness_of_fit,
        relative_error,
    )
    from src.metrics import (
        calculate_auc,
        calculate_auc24,
        calculate_auc_mic,
        calculate_auc_window,
        calculate_cmax,
        calculate_half_life,
        calculate_tmax,
        calculate_trough,
        trapezoid_auc,
    )
    from src.models import PKPDParams, auc_mic_response, one_compartment_ode, two_compartment_ode
    from src.simulation import (
        run_pipeline,
        simulate_one_compartment,
        simulate_repeated_regimen,
        simulate_two_compartment,
    )

    params = PKPDParams()
    assert one_compartment_ode(0.0, np.array([100.0]), 5.0, 10.0).shape == (1,)
    assert two_compartment_ode(0.0, np.array([100.0, 0.0]), 5.0, 2.0, 10.0, 20.0).shape == (2,)
    assert auc_mic_response(5.0, 1.0, 10.0, 10.0) == pytest.approx(3.5)

    one = simulate_one_compartment(100.0, 5.0, 10.0)
    two = simulate_two_compartment(100.0, params)
    pipeline = run_pipeline(100.0, params)
    repeated = simulate_repeated_regimen(1000.0, 12.0, params)
    assert one["concentration"][0] == pytest.approx(10.0)
    assert two["concentration"][0] == pytest.approx(10.0)
    assert pipeline["auc_0_48"] > pipeline["auc_0_24"] > 0
    assert repeated["auc24"] > 0

    time = np.array([0.0, 2.0, 4.0])
    concentration = np.array([10.0, 6.0, 4.0])
    assert trapezoid_auc(time, concentration) == pytest.approx(26.0)
    assert calculate_auc(time, concentration) == pytest.approx(26.0)
    assert calculate_auc_window(time, concentration, 0.0, 4.0) == pytest.approx(26.0)
    assert calculate_auc24(np.array([0.0, 24.0]), np.array([10.0, 10.0])) == pytest.approx(240.0)
    assert calculate_cmax(concentration) == 10.0
    assert calculate_tmax(time, concentration) == 0.0
    assert calculate_trough(concentration) == 4.0
    assert calculate_half_life(ke=0.5) == pytest.approx(np.log(2) / 0.5)
    assert calculate_auc_mic(500.0, 2.0) == 250.0

    observed = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 2.0, 3.0])
    assert np.all(absolute_error(observed, predicted) == 0)
    assert np.all(relative_error(observed, predicted) == 0)
    assert calculate_sse(observed, predicted) == 0.0
    assert calculate_rmse(observed, predicted) == 0.0
    assert calculate_mae(observed, predicted) == 0.0
    assert calculate_r_squared(observed, predicted) == 1.0
    assert calculate_mean_residual(observed, predicted) == 0.0
    assert calculate_aic(3, 1.0, 1) < 0
    assert calculate_bic(3, 1.0, 1) < 0
    assert goodness_of_fit(observed, predicted)["RMSE"] == 0.0


def test_data_fitting_optimization_and_population_are_callable():
    from src.data_processing import clean_data, flag_suspicious_rows, validate_pk_data
    from src.fitting import (
        bootstrap_fit,
        fit_multiple_subjects,
        fit_robust,
        fit_subject,
        parameter_standard_errors,
        predict_concentration,
        residuals,
        sum_squared_residuals,
        weighted_residuals,
    )
    from src.models import PKPDParams
    from src.optimization import (
        check_auc24_lower,
        check_auc24_upper,
        dosing_objective,
        is_regimen_feasible,
        optimize_from_starts,
        optimize_regimen,
        regimen_auc24,
        validate_regimen,
    )
    from src.population import (
        calculate_target_attainment,
        compare_regimens,
        generate_virtual_patients,
        lognormal_parameters,
        simulate_population,
    )

    data = pd.DataFrame(
        {
            "Subjects": [1, 1, 1, 1, 1, 1],
            "dose": [100, 0, 0, 0, 0, 0],
            "A1": [100, 80, 65, 50, 40, 30],
            "A2": [0, 10, 15, 18, 19, 19],
            "C": [10.0, 8.0, 6.5, 5.0, 4.0, 3.0],
            "Time": [0, 5, 10, 15, 20, 25],
            "Volume": [10] * 6,
            "CL": [0.5] * 6,
            "Q": [0.2] * 6,
        }
    )
    cleaned = clean_data(data)
    flagged = flag_suspicious_rows(cleaned)
    assert not flagged["Negative concentration"].any()
    assert validate_pk_data(cleaned)

    time = cleaned["Time"].to_numpy(dtype=float)
    concentration = cleaned["C"].to_numpy(dtype=float)
    theta = np.array([5.0, 2.0, 10.0, 20.0])
    prediction = predict_concentration(theta, time)
    assert prediction.shape == concentration.shape
    assert residuals(theta, time, concentration).shape == concentration.shape
    assert weighted_residuals(theta, time, concentration).shape == concentration.shape
    assert sum_squared_residuals(theta, time, concentration) >= 0
    result = fit_subject(time, concentration, initial_guess=theta)
    assert result.success
    assert parameter_standard_errors(result).shape == (4,)
    assert len(fit_multiple_subjects(cleaned, initial_guess=theta)) == 1
    assert len(fit_robust(cleaned, initial_guess=theta)) == 1
    assert len(bootstrap_fit(cleaned, 1, seed=42, initial_guess=theta)) == 1

    params = PKPDParams()
    assert regimen_auc24(1250.0, 12.0, params.CL) > 0
    assert dosing_objective([1250.0, 12.0], params) >= 0
    assert check_auc24_lower(500.0, params)
    assert check_auc24_upper(500.0, params)
    assert is_regimen_feasible(1250.0, 12.0, params)
    assert optimize_regimen([600.0, 20.0], params).success
    assert len(optimize_from_starts(params=params)) == 2
    assert validate_regimen(1000.0, 12.0, params)["auc24"] > 0

    mu, sigma = lognormal_parameters(5.0, 0.30)
    assert np.isfinite(mu) and sigma > 0
    patients = generate_virtual_patients(n=4, seed=42)
    population = simulate_population(patients, 1000.0, 12.0, include_profiles=False)
    assert len(population["results"]) == 4
    assert 0 <= calculate_target_attainment(population["results"]["AUC24"]) <= 1
    assert len(compare_regimens(patients, [(1000.0, 12.0), (1250.0, 12.0)])) == 2


def test_sensitivity_functions_are_callable_without_notebooks():
    from src.models import PKPDParams
    from src.sensitivity import (
        DEFAULT_PROBLEM,
        analyze_multiple_outputs,
        calculate_interaction_contribution,
        calculate_local_sensitivity,
        evaluate_parameter_samples,
        generate_sobol_samples,
        perturb_parameter,
        rank_local_sensitivity,
        rank_sensitivity_indices,
        run_sobol_analysis,
    )

    changed = perturb_parameter(PKPDParams(), "CL")
    assert changed.CL == pytest.approx(5.5)
    local = calculate_local_sensitivity()
    assert len(rank_local_sensitivity(local)) == 4
    samples = np.array([[5.0, 2.0, 10.0, 20.0], [6.0, 2.5, 11.0, 22.0]])
    outputs = evaluate_parameter_samples(samples)
    assert set(outputs) == {"auc_0_48", "auc_0_24", "cmax", "trough"}

    if importlib.util.find_spec("SALib") is None:
        with pytest.raises(ImportError):
            generate_sobol_samples(DEFAULT_PROBLEM, n_base=2)
        with pytest.raises(ImportError):
            run_sobol_analysis(DEFAULT_PROBLEM, outputs["auc_0_48"])
    else:
        sobol_samples = generate_sobol_samples(DEFAULT_PROBLEM, n_base=2, seed=42)
        sobol_outputs = evaluate_parameter_samples(sobol_samples)
        result = run_sobol_analysis(DEFAULT_PROBLEM, sobol_outputs["auc_0_48"])
        multiple = analyze_multiple_outputs(DEFAULT_PROBLEM, sobol_outputs)
        assert set(multiple) == set(sobol_outputs)
        assert len(calculate_interaction_contribution(result)) == 4
        assert len(rank_sensitivity_indices(DEFAULT_PROBLEM, result)) == 4


def test_end_to_end_pkpd_pipeline_is_sensible():
    from src.models import PKPDParams
    from src.simulation import run_pipeline

    params = PKPDParams()

    low_dose_result = run_pipeline(100.0, params)
    high_dose_result = run_pipeline(200.0, params)

    # Simulation produced a valid concentration-time profile.
    assert len(low_dose_result["time"]) == len(
        low_dose_result["concentration"]
    )
    assert low_dose_result["concentration"][0] == 100.0 / params.V1
    assert (low_dose_result["concentration"] >= 0).all()

    # PK → exposure → AUC/MIC → PD calculations are internally consistent.
    assert low_dose_result["auc_0_48"] > 0
    assert low_dose_result["auc_mic"] == pytest.approx(
        low_dose_result["auc_0_48"] / params.MIC
    )
    assert params.E0 <= low_dose_result["effect"] <= params.E0 + params.Emax

    # A larger dose should create larger exposure and a larger PD effect.
    assert high_dose_result["auc_0_48"] > low_dose_result["auc_0_48"]
    assert high_dose_result["auc_mic"] > low_dose_result["auc_mic"]
    assert high_dose_result["effect"] > low_dose_result["effect"]


def test_pipeline_rejects_malformed_negative_dose():
    from src.models import PKPDParams
    from src.simulation import run_pipeline

    with pytest.raises(ValueError, match="dose cannot be negative"):
        run_pipeline(-100.0, PKPDParams())


def test_pipeline_rejects_all_invalid_inputs_with_value_errors():
    from src.models import PKPDParams
    from src.simulation import run_pipeline, simulate_repeated_regimen

    invalid_cases = [
        (
            lambda: run_pipeline(-100.0, PKPDParams()),
            "dose cannot be negative",
        ),
        (
            lambda: run_pipeline(100.0, PKPDParams(CL=-1.0)),
            "CL cannot be negative",
        ),
        (
            lambda: run_pipeline(100.0, PKPDParams(Q=-1.0)),
            "Q cannot be negative",
        ),
        (
            lambda: run_pipeline(100.0, PKPDParams(V1=0.0)),
            "V1 must be strictly greater than zero",
        ),
        (
            lambda: run_pipeline(100.0, PKPDParams(V2=0.0)),
            "V2 must be strictly greater than zero",
        ),
        (
            lambda: run_pipeline(100.0, PKPDParams(MIC=-2.0)),
            "MIC must be strictly greater than zero",
        ),
        (
            lambda: simulate_repeated_regimen(
                1000.0, 0.0, PKPDParams()
            ),
            "dose, tau, and total_time must be greater than zero",
        ),
        (
            lambda: simulate_repeated_regimen(
                1000.0, 12.0, PKPDParams(), total_time=0.0
            ),
            "dose, tau, and total_time must be greater than zero",
        ),
    ]

    for invalid_call, expected_message in invalid_cases:
        with pytest.raises(ValueError, match=expected_message):
            invalid_call()

def test_full_pipeline_script_is_reproducible():
    from contextlib import redirect_stdout
    from io import StringIO
    from pathlib import Path
    import runpy

    script_path = (
        Path(__file__).resolve().parent.parent / "run_full_pipeline.py"
    )

    def run_script_and_capture_output():
        output = StringIO()
        with redirect_stdout(output):
            runpy.run_path(str(script_path), run_name="__main__")
        return output.getvalue()

    first_run_output = run_script_and_capture_output()
    second_run_output = run_script_and_capture_output()

    assert first_run_output == second_run_output
    assert "Recommended dose:" in first_run_output
    assert "POPULATION TARGET ATTAINMENT" in first_run_output
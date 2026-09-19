from src.models import PKPDParams
from src.optimization import optimize_from_starts, validate_regimen
from src.population import compare_regimens, generate_virtual_patients
from src.simulation import run_pipeline


def main():
    params = PKPDParams()

    # 1. Single-dose PK → exposure → AUC/MIC → illustrative PD effect
    single_dose_result = run_pipeline(dose=1000.0, params=params)

    print("SINGLE-DOSE PIPELINE")
    print(f"AUC 0–48: {single_dose_result['auc_0_48']:.2f}")
    print(f"AUC/MIC: {single_dose_result['auc_mic']:.2f}")
    print(f"Illustrative effect: {single_dose_result['effect']:.2f}")

    # 2. Optimize dose and interval from two starting points
    optimization_results = optimize_from_starts(params=params)

    for index, result in enumerate(optimization_results, start=1):
        dose_opt, tau_opt = result.x
        print(f"\nOPTIMIZER RUN {index}")
        print(f"Recommended dose: {dose_opt:.2f} mg")
        print(f"Recommended interval: q{tau_opt:.2f}h")

    # 3. Independently validate the first optimized regimen
    dose_opt, tau_opt = optimization_results[0].x
    validation = validate_regimen(dose_opt, tau_opt, params=params)

    print("\nREPEATED-REGIMEN VALIDATION")
    print(f"Simulated AUC24: {validation['auc24']:.2f}")
    print(f"Simulated AUC24/MIC: {validation['simulated_auc24_mic']:.2f}")

    # 4. Generate a reproducible virtual population and compare regimens
    patients = generate_virtual_patients(n=1000, seed=42)

    comparisons = compare_regimens(
        patients,
        regimens=[(1000.0, 12.0), (1250.0, 12.0)],
        include_profiles=False,
    )

    print("\nPOPULATION TARGET ATTAINMENT")
    for (dose, tau), result in comparisons.items():
        print(f"{dose:.0f} mg q{tau:.0f}h PTA: {result['pta']:.3f}")


if __name__ == "__main__":
    main()
# src/evaluation/gating.py
import argparse
import mlflow

# --- thresholds (tunable) ---
RULES = {
    "rmse_improvement_pct": 10,   # min % improvement over baseline RMSE
    "mape": 20,                  # max allowed MAPE
    "mase": 1.0                  # must be <= 1 (better than naive)
}

def check_gating(run_id: str):
    """
    Apply gating rules to an MLflow run.
    Reads metrics from the run and decides if it passes.
    """
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    print("📊 Retrieved metrics:", metrics)

    passed = True
    reasons = []

    # Rule 1: RMSE improvement vs baseline
    rel_improvement = metrics.get("rmse_improvement_pct", 0)
    if rel_improvement < RULES["rmse_improvement_pct"]:
        passed = False
        reasons.append(
            f"RMSE improvement {rel_improvement:.2f}% "
            f"< {RULES['rmse_improvement_pct']}%"
        )

    # Rule 2: MAPE threshold
    mape = metrics.get("mape", float("inf"))
    if mape > RULES["mape"]:
        passed = False
        reasons.append(f"MAPE {mape:.2f} > {RULES['mape']}")

    # Rule 3: MASE threshold
    mase = metrics.get("mase", float("inf"))
    if mase > RULES["mase"]:
        passed = False
        reasons.append(f"MASE {mase:.2f} > {RULES['mase']}")

    # Set gating result as a tag in MLflow
    client.set_tag(run_id, "gating", "pass" if passed else "fail")

    # --- Print summary ---
    print("\n🔎 Gating Evaluation")
    print(f"Run ID: {run_id}")
    print("-" * 40)
    print(f"{'Metric':<20} {'Value':<12} {'Threshold':<12} {'Pass?':<6}")
    print("-" * 40)
    print(f"{'rmse_improvement_pct':<20} {rel_improvement:<12.2f} >={RULES['rmse_improvement_pct']:<12} {rel_improvement >= RULES['rmse_improvement_pct']}")
    print(f"{'mape':<20} {mape:<12.2f} <={RULES['mape']:<12} {mape <= RULES['mape']}")
    print(f"{'mase':<20} {mase:<12.2f} <={RULES['mase']:<12} {mase <= RULES['mase']}")
    print("-" * 40)

    if passed:
        print(f"✅ Gating PASSED for run {run_id}")
    else:
        print(f"❌ Gating FAILED for run {run_id}")
        if reasons:
            print("Reasons:")
            for r in reasons:
                print(" -", r)

    return passed, reasons


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True, help="MLflow run_id to evaluate")
    args = parser.parse_args()

    check_gating(args.run_id)

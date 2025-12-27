from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from scipy import stats

from ml_training.features import FeatureSet2


def check_data_drift(df_reference, df_current, save_html: str = None):
    """
    Check for data drift between reference and current data using
    standard statistical tests (KS-Test for numeric, TVD for categorical).
    """
    drift_results = {}
    drifted_columns = []

    # Only check columns present in both datasets
    common_cols = [c for c in df_reference.columns if c in df_current.columns]

    for col in common_cols:
        # Check data type
        is_numeric = pd.api.types.is_numeric_dtype(df_reference[col])

        # 1. Calculate Drift Score
        if is_numeric:
            # Kolmogorov-Smirnov Test
            # The statistic (0 to 1) represents the max distance between CDFs.
            # Higher value = more drift.
            ks_stat, p_value = stats.ks_2samp(df_reference[col].dropna(), df_current[col].dropna())
            drift_score = ks_stat
            method = "KS-Test"
        else:
            # Total Variation Distance (TVD) for Categorical
            # Sum of absolute differences in probabilities / 2.
            # Range is 0 to 1.
            ref_counts = df_reference[col].value_counts(normalize=True)
            curr_counts = df_current[col].value_counts(normalize=True)

            # Union of all categories found in either set
            all_cats = set(ref_counts.index) | set(curr_counts.index)

            sum_abs_diff = sum(
                abs(ref_counts.get(cat, 0) - curr_counts.get(cat, 0)) for cat in all_cats
            )
            drift_score = sum_abs_diff / 2
            method = "TVD (Categorical)"

        # 2. Determine if Drifted (Threshold > 0.15 matches your previous logic)
        is_drifted = drift_score > 0.15

        if is_drifted:
            drifted_columns.append(col)

        drift_results[col] = {
            "Drift score": round(drift_score, 4),
            "Method": method,
            "Status": "Drift Detected" if is_drifted else "Pass",
        }

    # 3. Save HTML Report (Simple HTML Table)
    if save_html:
        results_df = pd.DataFrame(drift_results).T
        results_df.index.name = "Feature"

        html_content = f"""
        <html>
        <head><style>
            table {{ border-collapse: collapse; width: 100%; font-family: sans-serif; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
            th {{ background-color: #4CAF50; color: white; }}
            .drift {{ color: red; font-weight: bold; }}
        </style></head>
        <body>
            <h2>Data Drift Report</h2>
            <p>Reference Size: {len(df_reference)} | Current Size: {len(df_current)}</p>
            {results_df.to_html(classes="table")}
        </body>
        </html>
        """
        with open(save_html, "w", encoding="utf-8") as f:
            f.write(html_content)

    # 4. Return Summary
    n_columns = len(common_cols)
    n_drifted = len(drifted_columns)
    drift_share = n_drifted / n_columns if n_columns > 0 else 0.0

    return {
        "drift_detected": n_drifted > 0,
        "drift_share": drift_share,
        "n_drifted_columns": n_drifted,
        "n_columns": n_columns,
        "drifted_columns": drifted_columns,
    }


def main():
    # Load data
    data_path = "data_clean_process/customer_churn_cleaned.json"
    print(f"Loading data from {data_path}...")

    feature_gen = FeatureSet2(data_path)
    feature_gen.load_data()

    script_dir = Path(__file__).resolve().parent
    reports_dir = script_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Settings
    start_date = datetime(2018, 10, 29)
    periods = 4
    period_features = []

    # 1. Generate features for all periods
    for i in range(periods):
        cutoff = start_date + timedelta(days=i * 7)
        print(f"Generating Period {i + 1}: {cutoff.date()}")

        df = feature_gen.generate_features(
            cutoff_date=cutoff,
            observation_window_days=28,
            active_users_only=True,
        )

        # Drop ID column for drift check if present
        if "userId" in df.columns:
            df = df.drop(columns=["userId"])

        period_features.append(df)

    # 2. Compare Period 2, 3, 4 vs Period 1 (Reference)
    reference = period_features[0]
    print(f"\nReference Dataset: Period 1 ({start_date.date()})")

    for i in range(1, len(period_features)):
        current = period_features[i]
        period_num = i + 1

        print(f"\n--- Period {period_num} vs Reference ---")
        report_name = reports_dir / f"drift_report_period_{period_num}_vs_reference.html"

        # Run drift check
        result = check_data_drift(reference, current, save_html=report_name)

        # Print summary
        status = "DRIFT DETECTED!" if result["drift_detected"] else "No Drift"
        print(f"Status:      {status}")
        print(
            f"Drift Share: {result['drift_share']:.1%} ({result['n_drifted_columns']}/{result['n_columns']} columns)"
        )

        if result["drifted_columns"]:
            print(f"Drifted Cols: {', '.join(result['drifted_columns'][:3])}...")

        print(f"Report saved: {report_name}")


if __name__ == "__main__":
    main()

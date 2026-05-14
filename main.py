from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.pipeline import FraudDetectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud/spam message analysis pipeline")

    # ── SMS dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--sms-data", type=Path, required=True,
                        help="Path to Fraud SMS CSV dataset")
    parser.add_argument("--sms-text-col", default="text",
                        help="Column containing SMS message text (default: 'text')")
    parser.add_argument("--sms-label-col", default="label",
                        help="Column containing SMS class labels (default: 'label')")

    # ── Email dataset ─────────────────────────────────────────────────────────
    parser.add_argument("--email-data", type=Path, required=True,
                        help="Path to Phishing Email Corpus CSV dataset")
    parser.add_argument("--email-text-col", default="Email Text",
                        help="Column containing email body text (default: 'Email Text')")
    parser.add_argument("--email-label-col", default="Email Type",
                        help="Column containing email class labels (default: 'Email Type')")

    # ── Pipeline options ──────────────────────────────────────────────────────
    parser.add_argument("--n-clusters", type=int, default=5,
                        help="Number of K-Means++ clusters (default: 5)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional output JSON path")

    # ── Optimization options ──────────────────────────────────────────────────
    parser.add_argument("--n-estimators", type=int, default=300,
                        help="Number of Random Forest trees (default: 300)")
    parser.add_argument("--max-features", type=int, default=5000,
                        help="TF-IDF max features (default: 5000)")
    parser.add_argument("--min-support", type=float, default=0.15,
                        help="Apriori min support (default: 0.15)")
    parser.add_argument("--sample-size", type=float, default=1.0,
                        help="Fraction of data to use (default: 1.0 = 100%%)")
    parser.add_argument("--skip-rules", action="store_true",
                        help="Skip association rule mining")

    return parser.parse_args()


def load_sms(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    """Load Fraud SMS dataset and normalise to unified schema."""
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"SMS dataset must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  df[label_col].astype(str),
        "source": "sms",
    })
    return out.dropna(subset=["text"]).reset_index(drop=True)


def load_email(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    """Load Phishing Email Corpus and normalise to unified schema."""
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    if text_col not in df.columns or label_col not in df.columns:
        raise KeyError(
            f"Email dataset must contain columns '{text_col}' and '{label_col}'. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  df[label_col].astype(str),
        "source": "email",
    })
    return out.dropna(subset=["text"]).reset_index(drop=True)


def merge_datasets(sms_df: pd.DataFrame, email_df: pd.DataFrame) -> pd.DataFrame:
    """Concatenate both datasets and drop exact-duplicate messages."""
    combined = pd.concat([sms_df, email_df], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    after = len(combined)
    if before != after:
        print(f"[info] Stage 1/6: Removed {before - after} duplicate rows.")
    print(f"[info] SMS rows    : {len(sms_df)}")
    print(f"[info] Email rows  : {len(email_df)}")
    print(f"[info] Combined    : {after} rows")
    print(f"[info] Label counts:\n{combined['label'].value_counts().to_string()}")
    return combined


def main() -> None:
    args = parse_args()

    sms_df   = load_sms(args.sms_data,   args.sms_text_col,   args.sms_label_col)
    email_df = load_email(args.email_data, args.email_text_col, args.email_label_col)
    df       = merge_datasets(sms_df, email_df)
    
    # Apply sampling if requested
    if args.sample_size < 1.0:
        print(f"[info] Using {args.sample_size*100:.0f}% of data for quick testing")
        df = df.sample(frac=args.sample_size, random_state=42).reset_index(drop=True)
        print(f"[info] Sampled dataset: {len(df)} rows")

    pipeline = FraudDetectionPipeline(
        text_col="text",
        label_col="label",
        n_clusters=args.n_clusters,
        n_estimators=args.n_estimators,
        max_features=args.max_features,
        min_support=args.min_support,
        skip_rules=args.skip_rules,
    )
    summary = pipeline.fit(df)

    report = {
        "dataset_summary": {
            "sms_rows":    len(sms_df),
            "email_rows":  len(email_df),
            "total_rows":  len(df),
            "label_counts": df["label"].value_counts().to_dict(),
        },
        "evaluation":          summary.evaluation.report if summary.evaluation else {},
        "confusion_matrix":    summary.evaluation.confusion.tolist() if summary.evaluation else [],
        "roc_auc":             summary.evaluation.roc_auc if summary.evaluation else None,
        "top_features":        dict(list(summary.feature_importance.items())[:20]),
        "rules_found":         int(len(summary.rules)),
        "cluster_k":           summary.cluster_diagnostics.k_values if summary.cluster_diagnostics else [],
        "cluster_inertias":    summary.cluster_diagnostics.inertias if summary.cluster_diagnostics else [],
        "cluster_silhouettes": summary.cluster_diagnostics.silhouettes if summary.cluster_diagnostics else [],
    }

    # Save results JSON
    if args.output is not None:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[info] Results saved to {args.output}")
    
    # Save trained model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = models_dir / f"fraud_detection_model_{timestamp}.pkl"
    pipeline.save_model(model_file)
    
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

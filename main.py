from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.features import feature_names
from src.pipeline import FraudDetectionPipeline


CSV_ENCODINGS: tuple[str, ...] = ("utf-8", "utf-8-sig", "latin1", "cp1252")


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to read CSV file: {path}")


def pick_column(columns: Iterable[str], candidates: tuple[str, ...], dataset_name: str, kind: str) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    raise KeyError(
        f"{dataset_name} dataset must contain a {kind} column. Tried: {list(candidates)}. Found: {list(columns)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fraud/spam message analysis pipeline")

    # ── SMS dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--sms-data", type=Path, default=Path("data/sms/spam.csv"),
                        help="Path to Fraud SMS CSV dataset")
    parser.add_argument("--sms-text-col", default="text",
                        help="Column containing SMS message text (default: auto-detect)")
    parser.add_argument("--sms-label-col", default="label",
                        help="Column containing SMS class labels (default: auto-detect)")

    # ── Email dataset ─────────────────────────────────────────────────────────
    parser.add_argument("--email-data", type=Path, default=Path("data/email/phishing_email.csv"),
                        help="Path to Phishing Email Corpus CSV dataset")
    parser.add_argument("--email-text-col", default="Email Text",
                        help="Column containing email body text (default: auto-detect)")
    parser.add_argument("--email-label-col", default="Email Type",
                        help="Column containing email class labels (default: auto-detect)")

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
    df = read_csv_with_fallback(path)

    text_col = pick_column(df.columns, (text_col, "v2", "text", "message", "sms"), "SMS", "text")
    label_col = pick_column(df.columns, (label_col, "v1", "label", "class", "target"), "SMS", "label")

    out = pd.DataFrame({
        "text":   df[text_col].astype(str),
        "label":  df[label_col].astype(str),
        "source": "sms",
    })
    return out.dropna(subset=["text"]).reset_index(drop=True)


def load_email(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    """Load Phishing Email Corpus and normalise to unified schema."""
    df = read_csv_with_fallback(path)

    text_col = pick_column(df.columns, (text_col, "text_combined", "text", "body", "message"), "Email", "text")
    label_col = pick_column(df.columns, (label_col, "label", "type", "class", "target"), "Email", "label")

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


def generate_cluster_report(pipeline: FraudDetectionPipeline, df: pd.DataFrame) -> list[dict]:
    is_fraud = pipeline._coerce_labels(df["label"]) == 1
    fraud_df = df[is_fraud].copy()
    if fraud_df.empty:
        return []

    predictions = pipeline.predict(fraud_df["text"])
    fraud_df["cluster_id"] = predictions["cluster_id"].values
    
    features = feature_names(pipeline.artifacts.feature_bundle)
    centers = pipeline.artifacts.cluster_model.cluster_centers_
    
    cluster_reports = []
    # KMeans might not output exactly n_clusters if some are empty, but centers shape is (n_clusters, _)
    for cluster_id in range(centers.shape[0]):
        cluster_data = fraud_df[fraud_df["cluster_id"] == cluster_id]
        n_samples = len(cluster_data)
        if n_samples == 0:
            continue
            
        center = centers[cluster_id]
        top_indices = np.argsort(center)[::-1][:7]
        top_keywords = [features[i] for i in top_indices]
        
        examples = cluster_data["text"].sample(min(3, n_samples), random_state=42).tolist()
        
        suggested_name = f"Nhóm liên quan tới '{top_keywords[0]}'"
        
        cluster_reports.append({
            "cluster_id": int(cluster_id),
            "suggested_name": suggested_name,
            "n_samples": int(n_samples),
            "top_keywords": top_keywords,
            "examples": examples
        })
        
    return cluster_reports


def print_cluster_table(cluster_reports: list[dict]) -> None:
    print("\n" + "="*80)
    print("BẢNG MÔ TẢ CÁC CỤM (CLUSTERS)")
    print("="*80)
    for report in cluster_reports:
        print(f"Cụm {report['cluster_id']}: {report['suggested_name']}")
        print(f" - Số mẫu: {report['n_samples']}")
        print(f" - Từ khóa đặc trưng: {', '.join(report['top_keywords'])}")
        print(f" - Ví dụ tin nhắn:")
        for idx, ex in enumerate(report['examples']):
            ex_trunc = ex[:120].replace('\n', ' ') + "..." if len(ex) > 120 else ex.replace('\n', ' ')
            print(f"   {idx+1}. {ex_trunc}")
        print("-" * 80)
    print("\n")


def export_cluster_tables(cluster_reports: list[dict]) -> None:
    # Xuất ra Markdown
    md_lines = ["# Bảng Mô Tả Các Cụm (Clusters)\n"]
    for report in cluster_reports:
        md_lines.append(f"## Cụm {report['cluster_id']}: {report['suggested_name']}")
        md_lines.append(f"- **Số mẫu:** {report['n_samples']}")
        md_lines.append(f"- **Từ khóa đặc trưng:** {', '.join(report['top_keywords'])}")
        md_lines.append(f"- **Ví dụ tin nhắn:**")
        for idx, ex in enumerate(report['examples']):
            ex_trunc = ex[:200].replace('\n', ' ') + "..." if len(ex) > 200 else ex.replace('\n', ' ')
            md_lines.append(f"  {idx+1}. {ex_trunc}")
        md_lines.append("\n---\n")
    Path("cluster_descriptions.md").write_text("\n".join(md_lines), encoding="utf-8")
    
    # Xuất ra CSV
    csv_data = []
    for report in cluster_reports:
        csv_data.append({
            "Cụm": report["cluster_id"],
            "Tên gợi ý": report["suggested_name"],
            "Số mẫu": report["n_samples"],
            "Từ khóa đặc trưng": ", ".join(report["top_keywords"]),
            "Ví dụ 1": report["examples"][0] if len(report["examples"]) > 0 else "",
            "Ví dụ 2": report["examples"][1] if len(report["examples"]) > 1 else "",
            "Ví dụ 3": report["examples"][2] if len(report["examples"]) > 2 else "",
        })
    pd.DataFrame(csv_data).to_csv("cluster_descriptions.csv", index=False, encoding="utf-8-sig")
    print("[info] Đã lưu bảng mô tả cụm ra các tệp: cluster_descriptions.md và cluster_descriptions.csv")


def export_cluster_evaluation(pipeline: FraudDetectionPipeline, df: pd.DataFrame) -> None:
    is_fraud = pipeline._coerce_labels(df["label"]) == 1
    fraud_df = df[is_fraud].copy()
    if fraud_df.empty:
        return

    predictions = pipeline.predict(fraud_df["text"])
    labels = predictions["cluster_id"].values
    
    from src.preprocessing import preprocess_dataframe
    from src.features import transform_feature_bundle
    prepared = preprocess_dataframe(fraud_df, text_col="text")
    matrix = transform_feature_bundle(pipeline.artifacts.feature_bundle, prepared, text_col="clean_text")
    
    # Use dense matrix for metrics if possible, or sample if too large
    if matrix.shape[0] > 10000:
        print(f"[info] Lấy mẫu ngẫu nhiên 10000/{matrix.shape[0]} dòng để tính metric clustering...")
        np.random.seed(42)
        indices = np.random.choice(matrix.shape[0], 10000, replace=False)
        matrix = matrix[indices]
        labels = labels[indices]
    
    dense_matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix

    print("[info] Đang tính toán các chỉ số đánh giá cụm...")
    sil_score = silhouette_score(dense_matrix, labels)
    db_score = davies_bouldin_score(dense_matrix, labels)
    ch_score = calinski_harabasz_score(dense_matrix, labels)
    inertia = pipeline.artifacts.cluster_model.inertia_

    eval_data = [
        {"Chỉ số": "Silhouette Score", "Giá trị đạt được": round(sil_score, 4), "Ý nghĩa": "Càng gần 1 càng tốt"},
        {"Chỉ số": "Davies-Bouldin Index", "Giá trị đạt được": round(db_score, 4), "Ý nghĩa": "Càng nhỏ càng tốt"},
        {"Chỉ số": "Calinski-Harabasz Index", "Giá trị đạt được": round(ch_score, 4), "Ý nghĩa": "Càng lớn càng tốt"},
        {"Chỉ số": "Inertia (WCSS)", "Giá trị đạt được": round(inertia, 4), "Ý nghĩa": "Tổng phương sai nội cụm"},
    ]
    
    df_eval = pd.DataFrame(eval_data)
    
    print("\n" + "="*80)
    print("BẢNG ĐÁNH GIÁ CHẤT LƯỢNG PHÂN CỤM (CLUSTERING METRICS)")
    print("="*80)
    print(df_eval.to_markdown(index=False))
    print("="*80 + "\n")
    
    df_eval.to_csv("cluster_evaluation.csv", index=False, encoding="utf-8-sig")
    
    md_lines = [
        "# Bảng Đánh Giá Chất Lượng Phân Cụm\n",
        "Do phân cụm không giám sát, sử dụng các chỉ số nội tại để đánh giá:\n",
        df_eval.to_markdown(index=False),
        "\n"
    ]
    Path("cluster_evaluation.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("[info] Đã lưu bảng đánh giá ra: cluster_evaluation.md và cluster_evaluation.csv")


def plot_clusters(pipeline: FraudDetectionPipeline, df: pd.DataFrame, output_path: str = "cluster_visualization.png") -> None:
    is_fraud = pipeline._coerce_labels(df["label"]) == 1
    fraud_df = df[is_fraud].copy()
    if fraud_df.empty:
        return
        
    print("[info] Đang trích xuất đặc trưng để vẽ biểu đồ...")
    predictions = pipeline.predict(fraud_df["text"])
    cluster_ids = predictions["cluster_id"].values
    
    from src.preprocessing import preprocess_dataframe
    from src.features import transform_feature_bundle
    prepared = preprocess_dataframe(fraud_df, text_col="text")
    matrix = transform_feature_bundle(pipeline.artifacts.feature_bundle, prepared, text_col="clean_text")
    
    if matrix.shape[0] > 5000:
        print(f"[info] Lấy mẫu ngẫu nhiên 5000/{matrix.shape[0]} dòng để vẽ t-SNE...")
        np.random.seed(42)
        indices = np.random.choice(matrix.shape[0], 5000, replace=False)
        matrix = matrix[indices]
        cluster_ids = cluster_ids[indices]
        
    print("[info] Đang giảm chiều dữ liệu (SVD -> t-SNE)...")
    n_components = min(50, matrix.shape[1] - 1)
    if n_components > 2:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix_reduced = svd.fit_transform(matrix)
    else:
        matrix_reduced = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    coords_2d = tsne.fit_transform(matrix_reduced)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=coords_2d[:, 0], 
        y=coords_2d[:, 1],
        hue=cluster_ids,
        palette="tab10",
        legend="full",
        alpha=0.6
    )
    plt.title("t-SNE Visualization of Fraud Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Cluster ID")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[info] Đã lưu biểu đồ t-SNE vào: {output_path}")


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

    cluster_reports = generate_cluster_report(pipeline, df)
    report["cluster_descriptions"] = cluster_reports
    
    print_cluster_table(cluster_reports)
    export_cluster_tables(cluster_reports)
    export_cluster_evaluation(pipeline, df)
    plot_clusters(pipeline, df)

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

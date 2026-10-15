"""RFM-inspired value scores, K-Means/DBSCAN, and a 3-by-3 risk matrix.

Adapted from the supplied Week 8 script. Customer row alignment is checked
before the scorecard merge. TotalCharges is spending, not observed profit.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from project_paths import ROOT, output_dir, load_telco_source

def run(scorecard_path=None, seed=42):
    OUT = output_dir('segmentation')
    raw_df = load_telco_source()
    raw_df['TotalCharges'] = pd.to_numeric(raw_df['TotalCharges'], errors='coerce').fillna(0)
    raw_df = raw_df.reset_index(drop=True).reset_index().rename(columns={'index': 'customer_index'})
    path = scorecard_path or ROOT / 'results/reported/week7/customer_risk_scorecard.csv'
    risk_df = pd.read_csv(path)
    required = {'customer_index', 'churn_probability', 'risk_level'}
    if not required.issubset(risk_df):
        raise ValueError('The risk scorecard is missing required columns.')
    if risk_df.customer_index.duplicated().any() or set(risk_df.customer_index) != set(raw_df.customer_index):
        raise ValueError('The scorecard must contain each original customer_index exactly once.')
    if not risk_df.churn_probability.between(0, 1).all():
        raise ValueError('Risk scores must be probabilities in [0, 1].')
    if not risk_df.risk_level.isin(['Low Risk', 'Medium Risk', 'High Risk']).all():
        raise ValueError('Unexpected risk level in scorecard.')
    if 'customerID' in risk_df:
        check = raw_df[['customer_index', 'customerID']].merge(risk_df[['customer_index', 'customerID']], on='customer_index', validate='one_to_one')
        if not check.customerID_x.equals(check.customerID_y):
            raise ValueError('Customer IDs do not match the source row order.')
        risk_df = risk_df.drop(columns='customerID')
    df = raw_df.merge(risk_df, on='customer_index', how='left', validate='one_to_one')
    if 'actual_churn' in df and not df.actual_churn.eq((df.Churn == 'Yes').astype(int)).all():
        raise ValueError('Actual churn does not match the source rows; check row alignment.')
    df['churn_binary'] = (df.Churn == 'Yes').astype(int)

    # ------------------------------------------------------------
    # 3. Feature engineering for segmentation
    # ------------------------------------------------------------
    service_cols = [
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in service_cols:
        df[col + "_binary"] = (df[col] == "Yes").astype(int)

    # Frequency-like feature: number of service modules used by the customer
    df["service_count"] = df[[col + "_binary" for col in service_cols]].sum(axis=1)

    # Additional customer behavior/value features
    df["auto_payment"] = df["PaymentMethod"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]).astype(int)

    df["month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)


    df["R_relationship_tenure"] = df["tenure"]
    df["F_service_frequency"] = df["service_count"]
    df["M_total_monetary_value"] = df["TotalCharges"]

    segmentation_features = [
        "R_relationship_tenure",
        "F_service_frequency",
        "M_total_monetary_value",
    ]

    X_seg = df[segmentation_features]
    scaler = StandardScaler()
    X_seg_scaled = scaler.fit_transform(X_seg)


    # ------------------------------------------------------------
    # 4. Compare K-Means and DBSCAN
    # ------------------------------------------------------------
    comparison_rows = []

    # K-Means comparison with different k values
    for k in [3, 4, 5, 6]:
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = kmeans.fit_predict(X_seg_scaled)

        comparison_rows.append({
            "algorithm": "K-Means",
            "setting": f"k={k}",
            "n_clusters": len(set(labels)),
            "noise_rate": 0,
            "silhouette_score": silhouette_score(X_seg_scaled, labels),
            "calinski_harabasz_score": calinski_harabasz_score(X_seg_scaled, labels),
            "davies_bouldin_score": davies_bouldin_score(X_seg_scaled, labels),
        })

    # DBSCAN comparison with different eps values
    for eps in [0.35, 0.40, 0.45, 0.50]:
        dbscan = DBSCAN(eps=eps, min_samples=25)
        labels = dbscan.fit_predict(X_seg_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_rate = (labels == -1).mean()

        row = {
            "algorithm": "DBSCAN",
            "setting": f"eps={eps}, min_samples=25",
            "n_clusters": n_clusters,
            "noise_rate": noise_rate,
            "silhouette_score": np.nan,
            "calinski_harabasz_score": np.nan,
            "davies_bouldin_score": np.nan,
        }

        # Calculate clustering metrics
        valid_mask = labels != -1
        if n_clusters >= 2 and valid_mask.sum() > 0:
            row["silhouette_score"] = silhouette_score(X_seg_scaled[valid_mask], labels[valid_mask])
            row["calinski_harabasz_score"] = calinski_harabasz_score(X_seg_scaled[valid_mask], labels[valid_mask])
            row["davies_bouldin_score"] = davies_bouldin_score(X_seg_scaled[valid_mask], labels[valid_mask])

        comparison_rows.append(row)

    algorithm_comparison = pd.DataFrame(comparison_rows)

    # Final algorithm choice: K-Means k=4 gives the strongest and most interpretable result.
    final_kmeans = KMeans(n_clusters=4, random_state=seed, n_init=20)
    df["kmeans_cluster"] = final_kmeans.fit_predict(X_seg_scaled)

    final_dbscan = DBSCAN(eps=0.40, min_samples=25)
    df["dbscan_cluster"] = final_dbscan.fit_predict(X_seg_scaled)


    # ------------------------------------------------------------
    # 5. Build customer value score and value levels
    # ------------------------------------------------------------
    # Use percentile ranks so variables are comparable.
    df["monetary_score"] = df["TotalCharges"].rank(pct=True)
    df["tenure_score"] = df["tenure"].rank(pct=True)
    df["service_score"] = df["service_count"].rank(pct=True)

    # Weighted customer value score:
    # Monetary value is the most important, followed by tenure and service depth.
    df["customer_value_score"] = 100 * (
        0.50 * df["monetary_score"]
        + 0.30 * df["tenure_score"]
        + 0.20 * df["service_score"]
    )

    df["customer_value_level"] = pd.cut(
        df["customer_value_score"],
        bins=[-1, 40, 70, 101],
        labels=["Low Value", "Medium Value", "High Value"],
    )


    # ------------------------------------------------------------
    # 6. K-Means cluster summary
    # ------------------------------------------------------------
    cluster_summary = df.groupby("kmeans_cluster").agg(
        n_customers=("customerID", "count"),
        avg_tenure=("tenure", "mean"),
        avg_service_count=("service_count", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
        avg_total_charges=("TotalCharges", "mean"),
        total_value=("TotalCharges", "sum"),
        churn_rate=("churn_binary", "mean"),
        avg_churn_probability=("churn_probability", "mean"),
        avg_value_score=("customer_value_score", "mean"),
    ).reset_index()

    cluster_summary["customer_share"] = cluster_summary["n_customers"] / len(df)
    cluster_summary["value_share"] = cluster_summary["total_value"] / df["TotalCharges"].sum()

    def name_cluster(row):
        if row["avg_total_charges"] >= cluster_summary["avg_total_charges"].quantile(0.75):
            if row["avg_churn_probability"] >= cluster_summary["avg_churn_probability"].median():
                return "Core high-value customers with retention attention"
            else:
                return "Core high-value stable customers"
        elif row["avg_tenure"] <= cluster_summary["avg_tenure"].quantile(0.25):
            return "Early-stage low-value customers"
        elif row["avg_service_count"] <= cluster_summary["avg_service_count"].quantile(0.25):
            return "Basic-service medium-value customers"
        else:
            return "Growing medium-value customers"

    cluster_summary["business_segment_name"] = cluster_summary.apply(name_cluster, axis=1)

    cluster_name_map = dict(zip(cluster_summary["kmeans_cluster"], cluster_summary["business_segment_name"]))
    df["kmeans_segment_name"] = df["kmeans_cluster"].map(cluster_name_map)


    # ------------------------------------------------------------
    # 7. Churn-risk and customer-value matrix
    # ------------------------------------------------------------
    value_risk_matrix = df.groupby(["customer_value_level", "risk_level"], observed=True).agg(
        n_customers=("customerID", "count"),
        avg_churn_probability=("churn_probability", "mean"),
        actual_churn_rate=("churn_binary", "mean"),
        avg_value_score=("customer_value_score", "mean"),
        total_value=("TotalCharges", "sum"),
    ).reset_index()

    value_risk_matrix["customer_share"] = value_risk_matrix["n_customers"] / len(df)
    value_risk_matrix["value_share"] = value_risk_matrix["total_value"] / df["TotalCharges"].sum()

    # Sort by value contribution and calculate cumulative contribution.
    core_customer_groups = value_risk_matrix.sort_values("total_value", ascending=False).reset_index(drop=True)
    core_customer_groups["cumulative_value_share"] = core_customer_groups["value_share"].cumsum()
    core_customer_groups["is_core_60_percent_group"] = core_customer_groups["cumulative_value_share"] <= 0.60

    first_over_60 = core_customer_groups.index[core_customer_groups["cumulative_value_share"] >= 0.60]
    if len(first_over_60) > 0:
        core_customer_groups.loc[first_over_60[0], "is_core_60_percent_group"] = True

    core_60_groups = core_customer_groups[core_customer_groups["is_core_60_percent_group"]].copy()


    # ------------------------------------------------------------
    # 8. Export CSV outputs
    # ------------------------------------------------------------
    selected_columns = [
        "customer_index",
        "customerID",
        "tenure",
        "service_count",
        "MonthlyCharges",
        "TotalCharges",
        "churn_probability",
        "risk_level",
        "churn_binary",
        "R_relationship_tenure",
        "F_service_frequency",
        "M_total_monetary_value",
        "customer_value_score",
        "customer_value_level",
        "kmeans_cluster",
        "kmeans_segment_name",
        "dbscan_cluster",
    ]

    df[selected_columns].to_csv(OUT / "week8_customer_segmentation_results.csv", index=False, encoding="utf-8-sig")
    algorithm_comparison.to_csv(OUT / "week8_segmentation_algorithm_comparison.csv", index=False, encoding="utf-8-sig")
    cluster_summary.to_csv(OUT / "week8_kmeans_cluster_summary.csv", index=False, encoding="utf-8-sig")
    value_risk_matrix.to_csv(OUT / "week8_value_risk_matrix.csv", index=False, encoding="utf-8-sig")
    core_customer_groups.to_csv(OUT / "week8_core_customer_groups.csv", index=False, encoding="utf-8-sig")
    core_60_groups.to_csv(OUT / "week8_core_60_percent_customer_groups.csv", index=False, encoding="utf-8-sig")


    # ------------------------------------------------------------
    # 9. Black-and-white visualizations
    # ------------------------------------------------------------
    # Visualization 1: K-Means segmentation scatter plot
    plt.figure(figsize=(8, 6))
    markers = ["o", "s", "^", "x"]
    for cluster_id in sorted(df["kmeans_cluster"].unique()):
        part = df[df["kmeans_cluster"] == cluster_id]
        plt.scatter(
            part["TotalCharges"],
            part["tenure"],
            marker=markers[cluster_id % len(markers)],
            color="black",
            s=20,
            label=f"Cluster {cluster_id}",
            alpha=0.7,
        )
    plt.xlabel("TotalCharges")
    plt.ylabel("Tenure")
    plt.title("Week 8 K-Means Customer Segmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "week8_kmeans_customer_segmentation.png", dpi=300)
    plt.close()

    # Visualization 2: value-risk matrix contribution
    matrix_pivot = value_risk_matrix.pivot(
        index="customer_value_level",
        columns="risk_level",
        values="value_share",
    ).fillna(0)

    # Keep a stable order where possible
    value_order = ["High Value", "Medium Value", "Low Value"]
    risk_order = ["High Risk", "Medium Risk", "Low Risk"]
    matrix_pivot = matrix_pivot.reindex(index=value_order, columns=risk_order)

    plt.figure(figsize=(8, 5))
    for i, risk in enumerate(matrix_pivot.columns):
        plt.bar(
            matrix_pivot.index.astype(str),
            matrix_pivot[risk],
            bottom=matrix_pivot.iloc[:, :i].sum(axis=1) if i > 0 else None,
            label=risk,
            color=str(0.2 + i * 0.25),
            edgecolor="black",
        )
    plt.ylabel("Value Contribution Share")
    plt.title("Churn Risk and Customer Value Matrix")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "week8_value_risk_matrix.png", dpi=300)
    plt.close()

    # Visualization 3: core group cumulative contribution
    plot_core = core_customer_groups.copy()
    plot_core["group_name"] = plot_core["customer_value_level"].astype(str) + " / " + plot_core["risk_level"].astype(str)

    plt.figure(figsize=(10, 5))
    plt.plot(range(len(plot_core)), plot_core["cumulative_value_share"], marker="o", color="black")
    plt.axhline(y=0.60, color="black", linestyle="--", linewidth=1)
    plt.xticks(range(len(plot_core)), plot_core["group_name"], rotation=45, ha="right")
    plt.ylabel("Cumulative Value Share")
    plt.title("Core Customer Groups Contributing More Than 60% of Value")
    plt.tight_layout()
    plt.savefig(OUT / "week8_core_customer_value_contribution.png", dpi=300)
    plt.close()

    print("Week 8 analysis completed successfully.")
    print("Exported files:")
    print("- week8_customer_segmentation_results.csv")
    print("- week8_segmentation_algorithm_comparison.csv")
    print("- week8_kmeans_cluster_summary.csv")
    print("- week8_value_risk_matrix.csv")
    print("- week8_core_customer_groups.csv")
    print("- week8_core_60_percent_customer_groups.csv")
    print("- week8_kmeans_customer_segmentation.png")
    print("- week8_value_risk_matrix.png")
    print("- week8_core_customer_value_contribution.png")

    return OUT / "week8_customer_segmentation_results.csv"

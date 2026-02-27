"""
Quick Insights — Analytics Service (Análise Avançada)

Descriptive: central tendency, dispersion, position, histograms, correlation matrix,
             scatter plots, frequency tables + charts.
Predictive:  linear regression, logistic regression (AUC/KS/precision/recall/F1/accuracy/
             confusion matrix), KMeans clustering (silhouette, inertia).
All charts rendered via Chart.js.
"""

import json
import math
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, silhouette_score,
    explained_variance_score, calinski_harabasz_score, davies_bouldin_score,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, np.ndarray):
        return [_safe(x) for x in v]
    return v


def _data_to_df(data: dict) -> pd.DataFrame | None:
    rows = data.get("rows", [])
    if not rows:
        return None
    df = pd.DataFrame(rows)
    return df if not df.empty else None


# ---------------------------------------------------------------------------
# Descriptive Statistics
# ---------------------------------------------------------------------------

def compute_descriptive(data: dict) -> dict:
    df = _data_to_df(data)
    if df is None:
        return {"error": "Sem dados"}

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # --- Numeric stats ---
    numeric_stats = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        q1 = _safe(s.quantile(0.25))
        q2 = _safe(s.quantile(0.50))
        q3 = _safe(s.quantile(0.75))
        iqr = _safe(q3 - q1) if q1 is not None and q3 is not None else None
        mode_result = s.mode()
        mode_val = _safe(mode_result.iloc[0]) if not mode_result.empty else None
        numeric_stats.append({
            "column": col, "count": int(s.count()), "missing": int(df[col].isna().sum()),
            "mean": _safe(s.mean()), "median": _safe(s.median()), "mode": mode_val,
            "std": _safe(s.std()), "variance": _safe(s.var()),
            "min": _safe(s.min()), "max": _safe(s.max()), "range": _safe(s.max() - s.min()),
            "q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
            "skewness": _safe(s.skew()), "kurtosis": _safe(s.kurtosis()),
            "p5": _safe(s.quantile(0.05)), "p10": _safe(s.quantile(0.10)),
            "p90": _safe(s.quantile(0.90)), "p95": _safe(s.quantile(0.95)),
        })

    # --- Histograms ---
    histograms = {}
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty or len(s) < 2:
            continue
        counts, edges = np.histogram(s, bins=min(20, max(5, len(s) // 5)))
        histograms[col] = {
            "labels": [f"{_safe(edges[i])}" for i in range(len(counts))],
            "values": [int(c) for c in counts],
        }

    # --- Correlation matrix ---
    correlation = {}
    corr_cols = numeric_cols[:12]  # limit to 12 cols for readability
    if len(corr_cols) >= 2:
        corr_df = df[corr_cols].dropna()
        if len(corr_df) >= 3:
            corr_matrix = corr_df.corr(method="pearson")
            correlation = {
                "columns": corr_cols,
                "values": [[_safe(corr_matrix.iloc[i, j]) for j in range(len(corr_cols))] for i in range(len(corr_cols))],
            }

    # --- Frequency tables (categorical) ---
    freq_tables = {}
    for col in categorical_cols:
        vc = df[col].value_counts().head(30)
        freq_tables[col] = {
            "labels": vc.index.astype(str).tolist(),
            "values": vc.values.tolist(),
            "total": int(df[col].count()),
        }

    # --- Scatter pairs (first 4 numeric) ---
    scatter_pairs = []
    scatter_cols = numeric_cols[:4]
    for i in range(len(scatter_cols)):
        for j in range(i + 1, len(scatter_cols)):
            cx, cy = scatter_cols[i], scatter_cols[j]
            sample = df[[cx, cy]].dropna().head(200)
            if len(sample) < 2:
                continue
            scatter_pairs.append({
                "x_col": cx, "y_col": cy,
                "x": [_safe(v) for v in sample[cx].tolist()],
                "y": [_safe(v) for v in sample[cy].tolist()],
            })

    return {
        "row_count": len(df), "col_count": len(df.columns),
        "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
        "numeric_stats": numeric_stats, "histograms": histograms,
        "correlation": correlation,
        "freq_tables": freq_tables, "scatter_pairs": scatter_pairs,
    }


# ---------------------------------------------------------------------------
# KS Statistic
# ---------------------------------------------------------------------------

def _compute_ks(y_true, y_prob_positive):
    """Kolmogorov-Smirnov statistic for binary classification."""
    try:
        pos = y_prob_positive[y_true == 1]
        neg = y_prob_positive[y_true == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        stat, _ = sp_stats.ks_2samp(pos, neg)
        return _safe(stat)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Predictive Analysis
# ---------------------------------------------------------------------------

def run_prediction(data: dict, target: str, features: list[str], model_type: str) -> dict:
    df = _data_to_df(data)
    if df is None:
        return {"error": "Sem dados"}

    # --- Clustering (no target needed) ---
    if model_type == "clustering":
        return _run_clustering(df, features)

    if target not in df.columns:
        return {"error": f"Coluna alvo '{target}' não encontrada"}
    for f in features:
        if f not in df.columns:
            return {"error": f"Feature '{f}' não encontrada"}

    work = df[features + [target]].dropna()
    if len(work) < 10:
        return {"error": "Dados insuficientes (mínimo 10 registros sem nulos)"}

    # Encode categorical features
    encoders = {}
    X = work[features].copy()
    for col in features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = {str(v): int(i) for i, v in enumerate(le.classes_)}

    y = work[target].copy()

    if model_type == "logistic":
        return _run_logistic(X, y, features, target, work)
    else:
        return _run_linear(X, y, features, target, work)


def _classification_metrics(y_true, y_pred, y_prob=None):
    """Compute the 6 standard classification metrics. Works for binary/multiclass."""
    n_classes = len(set(y_true))
    is_binary = n_classes == 2
    avg = "binary" if is_binary else "weighted"

    metrics = {
        "accuracy": _safe(accuracy_score(y_true, y_pred)),
        "precision": _safe(precision_score(y_true, y_pred, average=avg, zero_division=0)),
        "recall": _safe(recall_score(y_true, y_pred, average=avg, zero_division=0)),
        "f1": _safe(f1_score(y_true, y_pred, average=avg, zero_division=0)),
        "auc": None,
        "ks": None,
    }

    if y_prob is not None:
        try:
            if is_binary:
                prob_pos = y_prob if y_prob.ndim == 1 else y_prob[:, 1]
                metrics["auc"] = _safe(roc_auc_score(y_true, prob_pos))
                metrics["ks"] = _compute_ks(y_true, prob_pos)
            else:
                metrics["auc"] = _safe(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            pass

    return metrics


def _run_linear(X, y, features, target, work):
    if not pd.api.types.is_numeric_dtype(y):
        return {"error": "Regressão linear requer coluna alvo numérica"}

    test_size = 0.2 if len(work) >= 50 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=test_size, random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # R² ajustado
    n = len(y_test)
    p = X_test.shape[1]
    r2 = r2_score(y_test, y_pred)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    try:
        mape = _safe(mean_absolute_percentage_error(y_test, y_pred))
    except Exception:
        mape = None

    # Classification metrics via binarization (above/below median)
    median_val = float(np.median(y_test))
    y_true_bin = (y_test > median_val).astype(int)
    y_pred_bin = (y_pred > median_val).astype(int)
    # Probability proxy: normalize distance from median to [0,1]
    y_range = max(y_pred.max() - y_pred.min(), 1e-9)
    y_prob_lin = (y_pred - y_pred.min()) / y_range
    clf_metrics = _classification_metrics(y_true_bin, y_pred_bin, y_prob_lin)

    return {
        "model_type": "linear", "target": target, "features": features,
        "metrics": {
            "r2": _safe(r2), "r2_adj": _safe(r2_adj),
            "mae": _safe(mean_absolute_error(y_test, y_pred)),
            "mse": _safe(mean_squared_error(y_test, y_pred)),
            "rmse": _safe(float(np.sqrt(mean_squared_error(y_test, y_pred)))),
            "mape": mape,
            "explained_var": _safe(explained_variance_score(y_test, y_pred)),
        },
        "classification_metrics": clf_metrics,
        "coefficients": {f: _safe(c) for f, c in zip(features, model.coef_)},
        "intercept": _safe(model.intercept_),
        "actual": [_safe(v) for v in y_test[:100]],
        "predicted": [_safe(v) for v in y_pred[:100]],
        "train_size": len(X_train), "test_size": len(X_test),
    }


def _run_logistic(X, y, features, target, work):
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y.astype(str))
    class_names = le_target.classes_.tolist()
    n_classes = len(class_names)
    is_binary = n_classes == 2

    test_size = 0.2 if len(work) >= 50 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_enc, test_size=test_size, random_state=42,
    )

    # Use lbfgs for multiclass (handles many classes better)
    try:
        model = LogisticRegression(
            max_iter=5000, random_state=42,
            solver="lbfgs", multi_class="multinomial" if n_classes > 2 else "auto",
        )
        model.fit(X_train, y_train)
    except Exception:
        # Fallback: saga solver with scaling
        try:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model = LogisticRegression(
                max_iter=5000, random_state=42,
                solver="saga", multi_class="multinomial" if n_classes > 2 else "auto",
            )
            model.fit(X_train_s, y_train)
            X_test = X_test_s
        except Exception as e:
            return {"error": f"Erro ao treinar modelo logístico: {str(e)[:200]}"}

    y_pred = model.predict(X_test)

    # Probabilities for AUC/KS
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        y_prob = None

    cm = confusion_matrix(y_test, y_pred)
    clf_metrics = _classification_metrics(y_test, y_pred, y_prob)

    # ROC curve data (binary only)
    roc_curve_data = None
    if is_binary and y_prob is not None:
        try:
            from sklearn.metrics import roc_curve
            prob_pos = y_prob[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob_pos)
            step = max(1, len(fpr) // 100)
            roc_curve_data = {
                "fpr": [_safe(v) for v in fpr[::step]],
                "tpr": [_safe(v) for v in tpr[::step]],
            }
        except Exception:
            pass

    return {
        "model_type": "logistic", "target": target, "features": features,
        "classification_metrics": clf_metrics,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "roc_curve": roc_curve_data,
        "train_size": len(X_train), "test_size": len(X_test),
    }


def _run_clustering(df, features):
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < 2:
        return {"error": "Clusterização requer pelo menos 2 features"}

    work = df[valid_features].dropna()
    # Encode categoricals
    X = work.copy()
    for col in valid_features:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    if len(X) < 10:
        return {"error": "Dados insuficientes (mínimo 10 registros sem nulos)"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # Find optimal k via elbow (2-8)
    max_k = min(8, len(X) // 3)
    if max_k < 2:
        max_k = 2
    inertias = []
    silhouettes = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append({"k": k, "inertia": _safe(km.inertia_)})
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0
        silhouettes.append({"k": k, "silhouette": _safe(sil)})

    # Best k = highest silhouette
    best_k = max(silhouettes, key=lambda s: s["silhouette"] or 0)["k"]

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = km_final.fit_predict(X_scaled)
    final_sil = _safe(silhouette_score(X_scaled, final_labels))

    # Additional clustering metrics
    try:
        calinski = _safe(calinski_harabasz_score(X_scaled, final_labels))
    except Exception:
        calinski = None
    try:
        davies = _safe(davies_bouldin_score(X_scaled, final_labels))
    except Exception:
        davies = None

    # Cluster sizes
    unique, counts = np.unique(final_labels, return_counts=True)
    cluster_sizes = [{"cluster": int(u), "size": int(c)} for u, c in zip(unique, counts)]

    # Cluster means (original scale)
    work_with_labels = work.copy()
    work_with_labels["_cluster"] = final_labels
    cluster_profiles = []
    for cl in sorted(unique):
        subset = work_with_labels[work_with_labels["_cluster"] == cl]
        profile = {"cluster": int(cl), "size": int(len(subset))}
        for f in valid_features:
            if pd.api.types.is_numeric_dtype(work[f]):
                profile[f] = _safe(subset[f].mean())
            else:
                profile[f] = str(subset[f].mode().iloc[0]) if not subset[f].mode().empty else ""
        cluster_profiles.append(profile)

    # Scatter data (first 2 features for 2D viz)
    f1, f2 = valid_features[0], valid_features[1]
    scatter = {
        "x_col": f1, "y_col": f2,
        "points": [
            {"x": _safe(X.iloc[i][f1]), "y": _safe(X.iloc[i][f2]), "c": int(final_labels[i])}
            for i in range(min(500, len(X)))
        ],
    }

    return {
        "model_type": "clustering", "features": valid_features,
        "best_k": best_k,
        "metrics": {
            "silhouette": final_sil,
            "inertia": _safe(km_final.inertia_),
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
        },
        "classification_metrics": {
            "accuracy": None, "precision": None, "recall": None,
            "f1": None, "auc": None, "ks": None,
        },
        "inertias": inertias, "silhouettes": silhouettes,
        "cluster_sizes": cluster_sizes, "cluster_profiles": cluster_profiles,
        "scatter": scatter, "total_points": len(X),
    }


# ---------------------------------------------------------------------------
# HTML Page Generator
# ---------------------------------------------------------------------------

def generate_analytics_html(data: dict) -> str:
    df = _data_to_df(data)
    if df is None:
        return _empty_html()

    desc = compute_descriptive(data)
    data_json = json.dumps(data, default=str)
    desc_json = json.dumps(desc, default=str)
    numeric_cols = desc["numeric_cols"]
    categorical_cols = desc["categorical_cols"]
    all_cols = numeric_cols + categorical_cols

    cols_json = json.dumps(all_cols)
    num_cols_json = json.dumps(numeric_cols)
    cat_cols_json = json.dumps(categorical_cols)

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quick Insights — Análise Avançada</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ background:#0a0c10; color:#c9d1d9; font-family:'Space Grotesk',sans-serif; }}
        ::-webkit-scrollbar {{ width:6px; height:6px; }}
        ::-webkit-scrollbar-track {{ background:transparent; }}
        ::-webkit-scrollbar-thumb {{ background:#30363d; border-radius:3px; }}

        .aa-header {{ background:#0d1117; border-bottom:1px solid #30363d; padding:12px 24px; display:flex; align-items:center; justify-content:space-between; position:sticky; top:0; z-index:100; }}
        .aa-logo {{ font-family:'JetBrains Mono',monospace; font-size:14px; font-weight:600; }}
        .aa-logo span {{ color:#ff6347; }}
        .aa-tabs {{ display:flex; gap:0; }}
        .aa-tab {{ padding:8px 20px; font-size:12px; font-weight:600; cursor:pointer; border:1px solid #30363d; background:#161b22; color:#8b949e; transition:all 0.15s; }}
        .aa-tab:first-child {{ border-radius:8px 0 0 8px; }}
        .aa-tab:last-child {{ border-radius:0 8px 8px 0; }}
        .aa-tab.active {{ background:#ff6347; color:white; border-color:#ff6347; }}
        .aa-tab:hover:not(.active) {{ color:#c9d1d9; background:#21262d; }}

        .aa-panel {{ display:none; padding:24px; max-width:1400px; margin:0 auto; }}
        .aa-panel.active {{ display:block; }}

        .aa-section {{ margin-bottom:28px; }}
        .aa-section-title {{ font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:600; color:#ff6347; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:12px; padding-bottom:8px; border-bottom:1px solid #21262d; }}

        .aa-grid {{ display:grid; gap:16px; }}
        .aa-grid-2 {{ grid-template-columns:repeat(auto-fit, minmax(500px, 1fr)); }}
        .aa-grid-3 {{ grid-template-columns:repeat(auto-fit, minmax(340px, 1fr)); }}
        .aa-grid-4 {{ grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); }}
        .aa-grid-6 {{ grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); }}

        .aa-card {{ background:#0d1117; border:1px solid #30363d; border-radius:12px; padding:16px; }}
        .aa-card-title {{ font-size:11px; font-weight:600; color:#58a6ff; margin-bottom:10px; font-family:'JetBrains Mono',monospace; text-transform:uppercase; letter-spacing:0.05em; }}

        .aa-stat {{ display:flex; justify-content:space-between; padding:4px 0; font-size:12px; border-bottom:1px solid #161b22; }}
        .aa-stat-label {{ color:#8b949e; }}
        .aa-stat-value {{ color:#c9d1d9; font-family:'JetBrains Mono',monospace; font-weight:500; }}

        .aa-chart-wrap {{ height:260px; position:relative; }}

        .aa-freq-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
        .aa-freq-table th {{ text-align:left; padding:6px 8px; background:#161b22; color:#ff6347; font-family:'JetBrains Mono',monospace; font-size:10px; text-transform:uppercase; letter-spacing:0.05em; border-bottom:1px solid #30363d; position:sticky; top:0; }}
        .aa-freq-table td {{ padding:5px 8px; border-bottom:1px solid #161b22; font-family:'JetBrains Mono',monospace; }}
        .aa-freq-table tr:hover td {{ background:#161b22; }}

        .aa-form-group {{ margin-bottom:14px; }}
        .aa-label {{ display:block; font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:4px; font-family:'JetBrains Mono',monospace; }}
        .aa-select {{ width:100%; background:#161b22; border:1px solid #30363d; color:#c9d1d9; padding:8px 12px; border-radius:8px; font-size:12px; font-family:'Space Grotesk',sans-serif; }}
        .aa-select:focus {{ border-color:#ff6347; outline:none; }}
        .aa-checkbox-list {{ max-height:180px; overflow-y:auto; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:8px; }}
        .aa-checkbox-item {{ display:flex; align-items:center; gap:6px; padding:3px 0; font-size:12px; cursor:pointer; }}
        .aa-checkbox-item input {{ accent-color:#ff6347; }}

        .aa-btn {{ padding:8px 20px; border-radius:8px; font-size:12px; font-weight:600; cursor:pointer; border:none; transition:all 0.15s; font-family:'Space Grotesk',sans-serif; }}
        .aa-btn-primary {{ background:#ff6347; color:white; }}
        .aa-btn-primary:hover {{ background:#ff4500; }}
        .aa-btn-primary:disabled {{ opacity:0.5; cursor:not-allowed; }}

        .aa-metric-card {{ background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px; text-align:center; }}
        .aa-metric-value {{ font-size:22px; font-weight:700; font-family:'JetBrains Mono',monospace; color:#39d353; }}
        .aa-metric-label {{ font-size:10px; color:#8b949e; text-transform:uppercase; letter-spacing:0.05em; margin-top:4px; }}

        .aa-coeff-bar {{ height:18px; border-radius:4px; min-width:2px; transition:width 0.4s; }}

        .aa-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:10px; font-family:'JetBrains Mono',monospace; font-weight:600; }}
        .aa-badge-num {{ background:rgba(88,166,255,0.15); color:#58a6ff; }}
        .aa-badge-cat {{ background:rgba(255,99,71,0.15); color:#ff6347; }}

        .aa-info {{ font-size:11px; color:#8b949e; background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px 14px; line-height:1.6; }}

        /* Correlation heatmap */
        .corr-grid {{ display:inline-grid; gap:1px; background:#21262d; border-radius:8px; overflow:hidden; }}
        .corr-cell {{ width:56px; height:32px; display:flex; align-items:center; justify-content:center; font-size:9px; font-family:'JetBrains Mono',monospace; font-weight:600; }}
        .corr-header {{ background:#161b22; color:#8b949e; font-size:8px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; padding:0 2px; }}
    </style>
</head>
<body>

<div class="aa-header">
    <div class="aa-logo">QUICK<span>INSIGHTS</span> — Análise Avançada</div>
    <div class="aa-tabs">
        <div class="aa-tab active" onclick="switchTab('descriptive',this)">Descritiva</div>
        <div class="aa-tab" onclick="switchTab('predictive',this)">Preditiva</div>
    </div>
    <div style="font-size:11px;color:#8b949e;font-family:'JetBrains Mono',monospace">
        {len(df)} registros · {len(df.columns)} colunas
    </div>
</div>

<!-- ==================== DESCRIPTIVE TAB ==================== -->
<div id="panel-descriptive" class="aa-panel active">

    <div class="aa-section">
        <div class="aa-section-title">Visão Geral do Dataset</div>
        <div class="aa-grid aa-grid-4">
            <div class="aa-metric-card"><div class="aa-metric-value">{len(df)}</div><div class="aa-metric-label">Registros</div></div>
            <div class="aa-metric-card"><div class="aa-metric-value">{len(numeric_cols)}</div><div class="aa-metric-label">Colunas Numéricas</div></div>
            <div class="aa-metric-card"><div class="aa-metric-value">{len(categorical_cols)}</div><div class="aa-metric-label">Colunas Categóricas</div></div>
            <div class="aa-metric-card"><div class="aa-metric-value">{int(df.isna().sum().sum())}</div><div class="aa-metric-label">Valores Nulos</div></div>
        </div>
    </div>

    <div class="aa-section" id="sectionNumericStats"></div>
    <div class="aa-section" id="sectionHistograms"></div>
    <div class="aa-section" id="sectionCorrelation"></div>
    <div class="aa-section" id="sectionScatter"></div>
    <div class="aa-section" id="sectionFrequency"></div>

</div>

<!-- ==================== PREDICTIVE TAB ==================== -->
<div id="panel-predictive" class="aa-panel">
    <div class="aa-grid" style="grid-template-columns:360px 1fr;gap:20px;">

        <div class="aa-card" style="position:sticky;top:70px;align-self:start;">
            <div class="aa-card-title">Configuração do Modelo</div>

            <div class="aa-form-group">
                <label class="aa-label">Tipo de Modelo</label>
                <select id="predModelType" class="aa-select" onchange="updatePredUI()">
                    <option value="linear">Regressão Linear</option>
                    <option value="logistic">Regressão Logística</option>
                    <option value="clustering">Clusterização (K-Means)</option>
                </select>
            </div>

            <div class="aa-info" id="predModelInfo" style="margin-bottom:14px;">
                Regressão Linear: prevê um valor numérico contínuo a partir das features selecionadas.
            </div>

            <div class="aa-form-group" id="predTargetGroup">
                <label class="aa-label">Variável Alvo (Y)</label>
                <select id="predTarget" class="aa-select">
                    {"".join(f'<option value="{c}">{c}</option>' for c in all_cols)}
                </select>
            </div>

            <div class="aa-form-group">
                <label class="aa-label">Features (X)</label>
                <div class="aa-checkbox-list" id="predFeatures">
                    {"".join(f'<label class="aa-checkbox-item"><input type="checkbox" value="{c}"><span>{c}</span> <span class="aa-badge {"aa-badge-num" if c in numeric_cols else "aa-badge-cat"}">{"num" if c in numeric_cols else "cat"}</span></label>' for c in all_cols)}
                </div>
            </div>

            <button class="aa-btn aa-btn-primary" style="width:100%" onclick="runPrediction()" id="predRunBtn">Executar Modelo</button>
            <div id="predStatus" style="margin-top:8px;font-size:11px;text-align:center;"></div>
        </div>

        <div id="predResult">
            <div class="aa-card" style="text-align:center;padding:40px;">
                <div style="color:#8b949e;font-size:13px;">Configure o modelo e clique em "Executar Modelo" para ver os resultados.</div>
            </div>
        </div>
    </div>
</div>

<script>
const DATA = {data_json};
const DESC = {desc_json};
const NUMERIC_COLS = {num_cols_json};
const CAT_COLS = {cat_cols_json};
const ALL_COLS = {cols_json};

Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#21262d';
Chart.defaults.font.family = "'Space Grotesk', sans-serif";

function switchTab(tab, el) {{
    document.querySelectorAll('.aa-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.aa-tab').forEach(t => t.classList.remove('active'));
    document.getElementById('panel-' + tab).classList.add('active');
    el.classList.add('active');
}}

// ============================
// Render Descriptive
// ============================
function renderNumericStats() {{
    const section = document.getElementById('sectionNumericStats');
    if (!DESC.numeric_stats || DESC.numeric_stats.length === 0) {{ section.innerHTML = ''; return; }}
    let html = '<div class="aa-section-title">Estatísticas Numéricas</div><div class="aa-grid aa-grid-3">';
    DESC.numeric_stats.forEach(s => {{
        html += `<div class="aa-card">
            <div class="aa-card-title">${{s.column}}</div>
            <div class="aa-stat"><span class="aa-stat-label">Média</span><span class="aa-stat-value">${{fmt(s.mean)}}</span></div>
            <div class="aa-stat"><span class="aa-stat-label">Mediana</span><span class="aa-stat-value">${{fmt(s.median)}}</span></div>
            <div class="aa-stat"><span class="aa-stat-label">Moda</span><span class="aa-stat-value">${{fmt(s.mode)}}</span></div>
            <div class="aa-stat"><span class="aa-stat-label">Desvio Padrão</span><span class="aa-stat-value">${{fmt(s.std)}}</span></div>
            <div class="aa-stat"><span class="aa-stat-label">Variância</span><span class="aa-stat-value">${{fmt(s.variance)}}</span></div>
            <div class="aa-stat"><span class="aa-stat-label">Amplitude</span><span class="aa-stat-value">${{fmt(s.range)}}</span></div>
            <div class="aa-stat"><span class="aa-stat-label">IQR (Q3-Q1)</span><span class="aa-stat-value">${{fmt(s.iqr)}}</span></div>
            <div style="margin-top:8px;padding-top:6px;border-top:1px solid #21262d;">
                <div class="aa-stat"><span class="aa-stat-label">Min</span><span class="aa-stat-value">${{fmt(s.min)}}</span></div>
                <div class="aa-stat"><span class="aa-stat-label">Q1 (25%)</span><span class="aa-stat-value">${{fmt(s.q1)}}</span></div>
                <div class="aa-stat"><span class="aa-stat-label">Q2 (50%)</span><span class="aa-stat-value">${{fmt(s.q2)}}</span></div>
                <div class="aa-stat"><span class="aa-stat-label">Q3 (75%)</span><span class="aa-stat-value">${{fmt(s.q3)}}</span></div>
                <div class="aa-stat"><span class="aa-stat-label">Max</span><span class="aa-stat-value">${{fmt(s.max)}}</span></div>
            </div>
            <div style="margin-top:6px;">
                <div class="aa-stat"><span class="aa-stat-label">Assimetria</span><span class="aa-stat-value">${{fmt(s.skewness)}}</span></div>
                <div class="aa-stat"><span class="aa-stat-label">Curtose</span><span class="aa-stat-value">${{fmt(s.kurtosis)}}</span></div>
                <div class="aa-stat"><span class="aa-stat-label">Nulos</span><span class="aa-stat-value">${{s.missing}}</span></div>
            </div>
        </div>`;
    }});
    html += '</div>';
    section.innerHTML = html;
}}

function renderHistograms() {{
    const section = document.getElementById('sectionHistograms');
    const hists = DESC.histograms;
    const cols = Object.keys(hists);
    if (cols.length === 0) {{ section.innerHTML = ''; return; }}
    let html = '<div class="aa-section-title">Histogramas</div><div class="aa-grid aa-grid-2">';
    cols.forEach((col, idx) => {{
        html += `<div class="aa-card"><div class="aa-card-title">${{col}}</div><div class="aa-chart-wrap"><canvas id="hist_${{idx}}"></canvas></div></div>`;
    }});
    html += '</div>';
    section.innerHTML = html;
    cols.forEach((col, idx) => {{
        new Chart(document.getElementById('hist_' + idx), {{
            type: 'bar',
            data: {{ labels: hists[col].labels, datasets: [{{ label: col, data: hists[col].values, backgroundColor: 'rgba(255,99,71,0.4)', borderColor: '#ff6347', borderWidth: 1, borderRadius: 3 }}] }},
            options: {{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ display:false }} }}, scales:{{ x:{{ ticks:{{ maxRotation:45, font:{{ size:9 }} }} }}, y:{{ beginAtZero:true }} }} }},
        }});
    }});
}}

function renderCorrelation() {{
    const section = document.getElementById('sectionCorrelation');
    const corr = DESC.correlation;
    if (!corr || !corr.columns || corr.columns.length < 2) {{ section.innerHTML = ''; return; }}

    const cols = corr.columns;
    const vals = corr.values;
    const n = cols.length;

    function corrColor(v) {{
        if (v === null || v === undefined) return '#161b22';
        const abs = Math.abs(v);
        if (v > 0) return `rgba(57, 211, 83, ${{(abs * 0.7 + 0.1).toFixed(2)}})`;
        return `rgba(255, 99, 71, ${{(abs * 0.7 + 0.1).toFixed(2)}})`;
    }}
    function corrText(v) {{
        if (v === null || v === undefined) return '—';
        return v.toFixed(2);
    }}

    let html = '<div class="aa-section-title">Matriz de Correlação (Pearson)</div>';
    html += '<div class="aa-card" style="overflow-x:auto;padding:20px;">';
    html += `<div class="corr-grid" style="grid-template-columns:80px repeat(${{n}}, 56px);">`;

    // Header row
    html += '<div class="corr-cell corr-header"></div>';
    cols.forEach(c => {{ html += `<div class="corr-cell corr-header" title="${{c}}">${{c.length > 7 ? c.slice(0,6) + '…' : c}}</div>`; }});

    // Data rows
    for (let i = 0; i < n; i++) {{
        html += `<div class="corr-cell corr-header" title="${{cols[i]}}" style="width:80px;justify-content:flex-end;padding-right:6px;">${{cols[i].length > 10 ? cols[i].slice(0,9) + '…' : cols[i]}}</div>`;
        for (let j = 0; j < n; j++) {{
            const v = vals[i][j];
            const bg = corrColor(v);
            const textColor = Math.abs(v) > 0.5 ? '#fff' : '#c9d1d9';
            html += `<div class="corr-cell" style="background:${{bg}};color:${{textColor}}" title="${{cols[i]}} × ${{cols[j]}}: ${{corrText(v)}}">${{corrText(v)}}</div>`;
        }}
    }}
    html += '</div>';

    // Legend
    html += `<div style="display:flex;align-items:center;gap:12px;margin-top:12px;font-size:10px;color:#8b949e;">
        <span>Legenda:</span>
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:14px;height:14px;border-radius:3px;background:rgba(255,99,71,0.7);"></span> Correlação negativa</span>
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:14px;height:14px;border-radius:3px;background:#161b22;border:1px solid #30363d;"></span> Sem correlação</span>
        <span style="display:flex;align-items:center;gap:4px;"><span style="width:14px;height:14px;border-radius:3px;background:rgba(57,211,83,0.7);"></span> Correlação positiva</span>
    </div>`;
    html += '</div>';
    section.innerHTML = html;
}}

function renderScatter() {{
    const section = document.getElementById('sectionScatter');
    const pairs = DESC.scatter_pairs;
    if (!pairs || pairs.length === 0) {{ section.innerHTML = ''; return; }}
    let html = '<div class="aa-section-title">Diagramas de Dispersão</div><div class="aa-grid aa-grid-2">';
    pairs.forEach((p, idx) => {{
        html += `<div class="aa-card"><div class="aa-card-title">${{p.x_col}} × ${{p.y_col}}</div><div class="aa-chart-wrap"><canvas id="scatter_${{idx}}"></canvas></div></div>`;
    }});
    html += '</div>';
    section.innerHTML = html;
    pairs.forEach((p, idx) => {{
        new Chart(document.getElementById('scatter_' + idx), {{
            type: 'scatter',
            data: {{ datasets: [{{ label: `${{p.x_col}} × ${{p.y_col}}`, data: p.x.map((x, i) => ({{ x, y: p.y[i] }})), backgroundColor: 'rgba(255,99,71,0.5)', borderColor: '#ff6347', pointRadius: 3, pointHoverRadius: 5 }}] }},
            options: {{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ display:false }} }}, scales:{{ x:{{ title:{{ display:true, text:p.x_col, color:'#c9d1d9', font:{{ size:11 }} }} }}, y:{{ title:{{ display:true, text:p.y_col, color:'#c9d1d9', font:{{ size:11 }} }} }} }} }},
        }});
    }});
}}

function renderFrequency() {{
    const section = document.getElementById('sectionFrequency');
    const ft = DESC.freq_tables;
    const cols = Object.keys(ft);
    if (cols.length === 0) {{ section.innerHTML = ''; return; }}
    const palette = ['#ff6347','#58a6ff','#39d353','#f0883e','#a371f7','#3fb950','#d2a8ff','#79c0ff','#ffa657','#ff7b72'];
    let html = '<div class="aa-section-title">Tabelas de Frequência + Gráficos</div><div class="aa-grid aa-grid-2">';
    cols.forEach((col, idx) => {{
        const f = ft[col];
        html += `<div class="aa-card">
            <div class="aa-card-title">${{col}} <span style="color:#8b949e;font-weight:400">(${{f.total}} registros)</span></div>
            <div class="aa-chart-wrap" style="height:200px;margin-bottom:12px;"><canvas id="freq_chart_${{idx}}"></canvas></div>
            <div style="max-height:180px;overflow-y:auto;">
                <table class="aa-freq-table"><thead><tr><th>Valor</th><th>Freq.</th><th>%</th></tr></thead>
                <tbody>${{f.labels.map((l, i) => `<tr><td>${{l}}</td><td>${{f.values[i]}}</td><td>${{(f.values[i]/f.total*100).toFixed(1)}}%</td></tr>`).join('')}}</tbody></table>
            </div>
        </div>`;
    }});
    html += '</div>';
    section.innerHTML = html;
    cols.forEach((col, idx) => {{
        const f = ft[col];
        const useBar = f.labels.length > 6;
        new Chart(document.getElementById('freq_chart_' + idx), {{
            type: useBar ? 'bar' : 'doughnut',
            data: {{ labels: f.labels, datasets: [{{ data: f.values, backgroundColor: useBar ? 'rgba(255,99,71,0.4)' : f.labels.map((_, i) => palette[i % palette.length]), borderColor: useBar ? '#ff6347' : '#0d1117', borderWidth: useBar ? 1 : 2, borderRadius: useBar ? 3 : 0 }}] }},
            options: {{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ display:!useBar, position:'right', labels:{{ font:{{ size:9 }} }} }} }}, ...(useBar ? {{ scales:{{ x:{{ ticks:{{ maxRotation:45, font:{{ size:9 }} }} }}, y:{{ beginAtZero:true }} }} }} : {{}}) }},
        }});
    }});
}}

// ============================
// Predictive UI
// ============================
function updatePredUI() {{
    const mt = document.getElementById('predModelType').value;
    const info = document.getElementById('predModelInfo');
    const targetGroup = document.getElementById('predTargetGroup');
    if (mt === 'linear') {{
        info.textContent = 'Regressão Linear: prevê um valor numérico contínuo. A variável alvo deve ser numérica.';
        targetGroup.style.display = 'block';
    }} else if (mt === 'logistic') {{
        info.textContent = 'Regressão Logística: classifica em categorias. Métricas: AUC, KS, Precision, Recall, F1, Acurácia + Matriz de Confusão.';
        targetGroup.style.display = 'block';
    }} else {{
        info.textContent = 'Clusterização K-Means: agrupa dados similares automaticamente. Não requer variável alvo. Selecione apenas as features.';
        targetGroup.style.display = 'none';
    }}
}}

async function runPrediction() {{
    const modelType = document.getElementById('predModelType').value;
    const target = document.getElementById('predTarget').value;
    const checkboxes = document.querySelectorAll('#predFeatures input:checked');
    const features = Array.from(checkboxes).map(cb => cb.value).filter(f => f !== target || modelType === 'clustering');

    if (features.length === 0) {{ alert('Selecione pelo menos uma feature.'); return; }}

    const btn = document.getElementById('predRunBtn');
    const status = document.getElementById('predStatus');
    btn.disabled = true;
    status.style.color = '#8b949e';
    status.textContent = 'Executando modelo...';

    try {{
        const body = {{ query_data: DATA, target: modelType === 'clustering' ? '' : target, features, model_type: modelType }};
        const res = await fetch('/api/analytics/predict', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify(body),
        }});
        const result = await res.json();
        if (result.error) {{
            status.style.color = '#ff6347';
            status.textContent = result.error;
            btn.disabled = false;
            return;
        }}
        status.textContent = '';
        renderPredictionResult(result);
    }} catch(e) {{
        status.style.color = '#ff6347';
        status.textContent = 'Erro: ' + e.message;
    }}
    btn.disabled = false;
}}

function renderPredictionResult(r) {{
    const container = document.getElementById('predResult');
    let html = '';

    if (r.model_type === 'linear') {{
        html += renderLinearResult(r);
    }} else if (r.model_type === 'logistic') {{
        html += renderLogisticResult(r);
    }} else if (r.model_type === 'clustering') {{
        html += renderClusteringResult(r);
    }}

    container.innerHTML = html;
    setTimeout(() => renderPredCharts(r), 50);
}}

function renderClfMetrics(clf, label) {{
    if (!clf) return '';
    const fmtPct = v => v !== null && v !== undefined ? (v * 100).toFixed(1) + '%' : '—';
    return `<div class="aa-section"><div class="aa-section-title">${{label || 'Métricas Estatísticas'}}</div>
    <div class="aa-grid aa-grid-6" style="margin-bottom:16px;">
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmtPct(clf.accuracy)}}</div><div class="aa-metric-label">Acurácia</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmtPct(clf.precision)}}</div><div class="aa-metric-label">Precision</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmtPct(clf.recall)}}</div><div class="aa-metric-label">Recall</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{clf.f1 !== null && clf.f1 !== undefined ? fmt(clf.f1) : '—'}}</div><div class="aa-metric-label">F1-Score</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{clf.auc !== null && clf.auc !== undefined ? fmt(clf.auc) : '—'}}</div><div class="aa-metric-label">AUC-ROC</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{clf.ks !== null && clf.ks !== undefined ? fmt(clf.ks) : '—'}}</div><div class="aa-metric-label">KS</div></div>
    </div></div>`;
}}

function renderLinearResult(r) {{
    const m = r.metrics;
    let html = `<div class="aa-section"><div class="aa-section-title">Regressão Linear — ${{r.target}}</div>
    <div class="aa-grid aa-grid-6" style="margin-bottom:16px;">
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.r2)}}</div><div class="aa-metric-label">R²</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.r2_adj)}}</div><div class="aa-metric-label">R² Ajustado</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.mae)}}</div><div class="aa-metric-label">MAE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.rmse)}}</div><div class="aa-metric-label">RMSE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{m.mape !== null ? (m.mape * 100).toFixed(1) + '%' : '—'}}</div><div class="aa-metric-label">MAPE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.explained_var)}}</div><div class="aa-metric-label">Var. Explicada</div></div>
    </div>
    <div class="aa-grid aa-grid-4" style="margin-bottom:16px;">
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.mse)}}</div><div class="aa-metric-label">MSE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{r.train_size}}/${{r.test_size}}</div><div class="aa-metric-label">Train/Test</div></div>
    </div></div>`;

    html += renderClfMetrics(r.classification_metrics, 'Métricas de Classificação (binarizado pela mediana)');

    const coeffs = Object.entries(r.coefficients);
    html += `<div class="aa-section"><div class="aa-section-title">Coeficientes</div><div class="aa-card">
        <div class="aa-stat" style="border-bottom:1px solid #30363d;margin-bottom:8px;"><span class="aa-stat-label" style="font-weight:600">Intercepto</span><span class="aa-stat-value">${{fmt(r.intercept)}}</span></div>`;
    const maxC = Math.max(...coeffs.map(([_,v]) => Math.abs(v)), 0.001);
    coeffs.forEach(([name, val]) => {{
        const pct = Math.abs(val) / maxC * 100;
        const color = val >= 0 ? '#39d353' : '#ff6347';
        html += `<div style="margin:6px 0;"><div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px;"><span class="aa-stat-label">${{name}}</span><span class="aa-stat-value" style="color:${{color}}">${{fmt(val)}}</span></div><div style="background:#161b22;border-radius:4px;height:18px;"><div class="aa-coeff-bar" style="width:${{pct}}%;background:${{color}}20;border:1px solid ${{color}}50;"></div></div></div>`;
    }});
    html += '</div></div>';
    html += `<div class="aa-section"><div class="aa-section-title">Real vs Previsto</div><div class="aa-card"><div style="height:300px;"><canvas id="predChart1"></canvas></div></div></div>`;
    return html;
}}

function renderLogisticResult(r) {{
    let html = `<div class="aa-section"><div class="aa-section-title">Regressão Logística — ${{r.target}}</div></div>`;

    html += renderClfMetrics(r.classification_metrics, 'Métricas Estatísticas');

    html += `<div class="aa-info" style="margin-bottom:16px;">Train: ${{r.train_size}} · Test: ${{r.test_size}}</div>`;

    // Confusion matrix + ROC
    html += `<div class="aa-section"><div class="aa-section-title">Matriz de Confusão & Curva ROC</div>
    <div class="aa-grid" style="grid-template-columns:1fr 1fr;gap:16px;">`;

    html += '<div class="aa-card"><div class="aa-card-title">Matriz de Confusão</div>';
    if (r.class_names && r.confusion_matrix) {{
        html += '<div style="overflow-x:auto;max-height:400px;overflow-y:auto;"><table class="aa-freq-table"><thead><tr><th style="min-width:80px;">Real \\ Pred</th>';
        r.class_names.forEach(c => html += `<th>${{c}}</th>`);
        html += '</tr></thead><tbody>';
        r.confusion_matrix.forEach((row, i) => {{
            html += `<tr><td style="font-weight:600;color:#58a6ff">${{r.class_names[i]}}</td>`;
            row.forEach((v, j) => {{
                const bg = i === j ? 'rgba(57,211,83,0.15)' : (v > 0 ? 'rgba(255,99,71,0.1)' : '');
                html += `<td style="background:${{bg}}">${{v}}</td>`;
            }});
            html += '</tr>';
        }});
        html += '</tbody></table></div>';
    }}
    html += '</div>';

    html += '<div class="aa-card"><div class="aa-card-title">Curva ROC</div>';
    if (r.roc_curve) {{
        html += '<div style="height:280px;"><canvas id="predChart1"></canvas></div>';
    }} else {{
        html += '<div style="padding:40px;text-align:center;color:#8b949e;font-size:12px;">Curva ROC disponível apenas para classificação binária.</div>';
    }}
    html += '</div></div></div>';

    html += `<div class="aa-section"><div class="aa-section-title">Distribuição das Predições</div><div class="aa-card"><div style="height:260px;"><canvas id="predChart2"></canvas></div></div></div>`;

    return html;
}}

function renderClusteringResult(r) {{
    const m = r.metrics;
    let html = `<div class="aa-section"><div class="aa-section-title">Clusterização K-Means — ${{r.best_k}} Clusters</div>
    <div class="aa-grid aa-grid-6" style="margin-bottom:16px;">
        <div class="aa-metric-card"><div class="aa-metric-value">${{r.best_k}}</div><div class="aa-metric-label">Clusters (K)</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.silhouette)}}</div><div class="aa-metric-label">Silhouette</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.inertia)}}</div><div class="aa-metric-label">Inertia</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.calinski_harabasz)}}</div><div class="aa-metric-label">Calinski-Harabasz</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.davies_bouldin)}}</div><div class="aa-metric-label">Davies-Bouldin</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{r.total_points}}</div><div class="aa-metric-label">Total Pontos</div></div>
    </div></div>`;

    html += renderClfMetrics(r.classification_metrics, 'Métricas Estatísticas (não aplicável — sem variável alvo)');

    html += '<div class="aa-section"><div class="aa-section-title">Tamanho dos Clusters</div><div class="aa-grid aa-grid-4">';
    const clPalette = ['#ff6347','#58a6ff','#39d353','#f0883e','#a371f7','#3fb950','#d2a8ff','#79c0ff'];
    r.cluster_sizes.forEach(cs => {{
        html += `<div class="aa-metric-card" style="border-left:3px solid ${{clPalette[cs.cluster % clPalette.length]}}"><div class="aa-metric-value">${{cs.size}}</div><div class="aa-metric-label">Cluster ${{cs.cluster}}</div></div>`;
    }});
    html += '</div></div>';

    html += `<div class="aa-section"><div class="aa-section-title">Visualizações</div>
    <div class="aa-grid aa-grid-2">
        <div class="aa-card"><div class="aa-card-title">Dispersão (${{r.scatter.x_col}} × ${{r.scatter.y_col}})</div><div style="height:300px;"><canvas id="predChart1"></canvas></div></div>
        <div class="aa-card"><div class="aa-card-title">Método do Cotovelo (Elbow)</div><div style="height:300px;"><canvas id="predChart2"></canvas></div></div>
    </div></div>`;

    if (r.cluster_profiles && r.cluster_profiles.length > 0) {{
        html += '<div class="aa-section"><div class="aa-section-title">Perfil dos Clusters (Médias)</div>';
        html += '<div class="aa-card" style="overflow-x:auto;"><table class="aa-freq-table"><thead><tr><th>Cluster</th><th>Tamanho</th>';
        r.features.forEach(f => html += `<th>${{f}}</th>`);
        html += '</tr></thead><tbody>';
        r.cluster_profiles.forEach(p => {{
            html += `<tr><td style="font-weight:600;color:${{clPalette[p.cluster % clPalette.length]}}">Cluster ${{p.cluster}}</td><td>${{p.size}}</td>`;
            r.features.forEach(f => html += `<td>${{fmt(p[f])}}</td>`);
            html += '</tr>';
        }});
        html += '</tbody></table></div></div>';
    }}

    return html;
}}

function renderPredCharts(r) {{
    if (r.model_type === 'linear') {{
        const c1 = document.getElementById('predChart1');
        if (c1) {{
            new Chart(c1, {{
                type: 'scatter',
                data: {{ datasets: [
                    {{ label: 'Real vs Previsto', data: r.actual.map((a, i) => ({{ x: a, y: r.predicted[i] }})), backgroundColor: 'rgba(255,99,71,0.5)', borderColor: '#ff6347', pointRadius: 3 }},
                    {{ label: 'Perfeito', data: [{{ x: Math.min(...r.actual), y: Math.min(...r.actual) }}, {{ x: Math.max(...r.actual), y: Math.max(...r.actual) }}], type: 'line', borderColor: '#39d353', borderDash: [5,5], pointRadius: 0, borderWidth: 2 }},
                ] }},
                options: {{ responsive:true, maintainAspectRatio:false, scales:{{ x:{{ title:{{ display:true, text:'Real', color:'#c9d1d9' }} }}, y:{{ title:{{ display:true, text:'Previsto', color:'#c9d1d9' }} }} }} }},
            }});
        }}

    }} else if (r.model_type === 'logistic') {{
        // ROC curve
        if (r.roc_curve) {{
            const c1 = document.getElementById('predChart1');
            if (c1) {{
                new Chart(c1, {{
                    type: 'line',
                    data: {{ labels: r.roc_curve.fpr, datasets: [
                        {{ label: `ROC (AUC=${{r.classification_metrics && r.classification_metrics.auc ? r.classification_metrics.auc.toFixed(3) : '—'}})`, data: r.roc_curve.tpr, borderColor: '#ff6347', backgroundColor: 'rgba(255,99,71,0.1)', fill: true, tension: 0.2, pointRadius: 0 }},
                        {{ label: 'Aleatório', data: r.roc_curve.fpr, borderColor: '#30363d', borderDash: [5,5], pointRadius: 0, borderWidth: 1 }},
                    ] }},
                    options: {{ responsive:true, maintainAspectRatio:false, scales:{{ x:{{ title:{{ display:true, text:'FPR (False Positive Rate)', color:'#c9d1d9', font:{{ size:10 }} }}, ticks:{{ callback: v => typeof v==='number' ? v.toFixed(1) : v }} }}, y:{{ title:{{ display:true, text:'TPR (True Positive Rate)', color:'#c9d1d9', font:{{ size:10 }} }}, min:0, max:1 }} }} }},
                }});
            }}
        }}

        // Confusion matrix bar
        const c2 = document.getElementById('predChart2');
        if (c2 && r.confusion_matrix) {{
            const cm = r.confusion_matrix;
            const labels = [];
            const values = [];
            const colors = [];
            r.class_names.forEach((real, i) => {{
                r.class_names.forEach((pred, j) => {{
                    labels.push(`R:${{real}} P:${{pred}}`);
                    values.push(cm[i][j]);
                    colors.push(i === j ? 'rgba(57,211,83,0.6)' : 'rgba(255,99,71,0.5)');
                }});
            }});
            new Chart(c2, {{
                type: 'bar',
                data: {{ labels, datasets: [{{ data: values, backgroundColor: colors, borderWidth: 0, borderRadius: 3 }}] }},
                options: {{ responsive:true, maintainAspectRatio:false, plugins:{{ legend:{{ display:false }} }}, scales:{{ x:{{ ticks:{{ font:{{ size:9 }} }} }}, y:{{ beginAtZero:true }} }} }},
            }});
        }}

    }} else if (r.model_type === 'clustering') {{
        const clPalette = ['#ff6347','#58a6ff','#39d353','#f0883e','#a371f7','#3fb950','#d2a8ff','#79c0ff'];

        // Scatter
        const c1 = document.getElementById('predChart1');
        if (c1 && r.scatter) {{
            const datasets = [];
            for (let k = 0; k < r.best_k; k++) {{
                const pts = r.scatter.points.filter(p => p.c === k).map(p => ({{ x: p.x, y: p.y }}));
                datasets.push({{ label: `Cluster ${{k}}`, data: pts, backgroundColor: clPalette[k % clPalette.length] + '80', borderColor: clPalette[k % clPalette.length], pointRadius: 3, pointHoverRadius: 5 }});
            }}
            new Chart(c1, {{
                type: 'scatter',
                data: {{ datasets }},
                options: {{ responsive:true, maintainAspectRatio:false, scales:{{ x:{{ title:{{ display:true, text:r.scatter.x_col, color:'#c9d1d9' }} }}, y:{{ title:{{ display:true, text:r.scatter.y_col, color:'#c9d1d9' }} }} }} }},
            }});
        }}

        // Elbow
        const c2 = document.getElementById('predChart2');
        if (c2 && r.inertias) {{
            new Chart(c2, {{
                type: 'line',
                data: {{
                    labels: r.inertias.map(d => 'K=' + d.k),
                    datasets: [
                        {{ label: 'Inertia', data: r.inertias.map(d => d.inertia), borderColor: '#ff6347', backgroundColor: 'rgba(255,99,71,0.1)', fill: true, tension: 0.3 }},
                        {{ label: 'Silhouette', data: r.silhouettes.map(d => d.silhouette), borderColor: '#39d353', yAxisID: 'y1', tension: 0.3 }},
                    ],
                }},
                options: {{ responsive:true, maintainAspectRatio:false, scales:{{ y:{{ title:{{ display:true, text:'Inertia', color:'#ff6347' }}, position:'left' }}, y1:{{ title:{{ display:true, text:'Silhouette', color:'#39d353' }}, position:'right', grid:{{ drawOnChartArea:false }}, min:0, max:1 }} }} }},
            }});
        }}
    }}
}}

// ============================
// Utils
// ============================
function fmt(v) {{
    if (v === null || v === undefined) return '—';
    if (typeof v === 'number') {{
        if (Number.isInteger(v)) return v.toLocaleString('pt-BR');
        return v.toLocaleString('pt-BR', {{ minimumFractionDigits: 2, maximumFractionDigits: 4 }});
    }}
    return String(v);
}}

// ============================
// Init
// ============================
renderNumericStats();
renderHistograms();
renderCorrelation();
renderScatter();
renderFrequency();
</script>

</body>
</html>"""


def _empty_html() -> str:
    return """<!DOCTYPE html>
<html><head><title>Quick Insights</title></head>
<body style="background:#0a0c10;color:#8b949e;font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0">
<div style="text-align:center">
<h2 style="color:#ff6347">Sem dados para análise</h2>
<p>Execute uma consulta que retorne resultados tabulares.</p>
</div></body></html>"""
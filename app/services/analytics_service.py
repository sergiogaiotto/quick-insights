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

def run_prediction(data: dict, target: str, features: list[str], model_type: str, n_clusters: int = 0) -> dict:
    df = _data_to_df(data)
    if df is None:
        return {"error": "Sem dados"}

    # --- Clustering (no target needed) ---
    if model_type == "clustering":
        return _run_clustering(df, features, n_clusters=n_clusters)

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

    # Label-encode categorical target
    target_encoded = False
    if not pd.api.types.is_numeric_dtype(y):
        le_y = LabelEncoder()
        y = pd.Series(le_y.fit_transform(y.astype(str)), index=y.index)
        target_encoded = True

    if model_type == "logistic":
        return _run_logistic(X, y, features, target, work)
    else:
        return _run_linear(X, y, features, target, work, target_encoded=target_encoded)


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


def _coeff_table_linear(X_train, y_train, model, features):
    """Compute coeff, SE, t(Wald), p-value, exp(b), 95% CI for linear regression."""
    n, p = X_train.shape
    df = n - p - 1
    if df < 1:
        return []

    try:
        y_pred_train = model.predict(X_train)
        residuals = y_train - y_pred_train
        mse = float(np.sum(residuals ** 2) / df)

        # Add intercept column
        X_with_int = np.column_stack([np.ones(n), X_train])
        XtX_inv = np.linalg.inv(X_with_int.T @ X_with_int)
        var_covar = XtX_inv * mse

        all_coefs = np.concatenate([[model.intercept_], model.coef_])
        all_names = ["(Intercepto)"] + list(features)
        se_all = np.sqrt(np.diag(var_covar))

        t_crit = sp_stats.t.ppf(0.975, df)
        table = []
        for i, name in enumerate(all_names):
            coef = float(all_coefs[i])
            se = float(se_all[i]) if se_all[i] > 0 else 1e-12
            t_val = coef / se
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_val), df))
            exp_b = math.exp(min(coef, 500))  # clamp to avoid overflow
            lower = coef - t_crit * se
            upper = coef + t_crit * se
            table.append({
                "name": name,
                "coeff": _safe(coef), "se": _safe(se),
                "wald": _safe(t_val), "p_value": _safe(p_val),
                "exp_b": _safe(exp_b),
                "lower": _safe(lower), "upper": _safe(upper),
                "significant": bool(p_val < 0.05),
            })
        return table
    except Exception:
        return []


def _coeff_table_logistic(X_train, y_train, model, features):
    """Compute coeff, SE, Wald, p-value, exp(b), 95% CI for logistic regression."""
    try:
        n_classes = len(model.classes_)
        # Use mean coefficients across classes for multiclass
        if model.coef_.ndim == 2 and model.coef_.shape[0] > 1:
            coefs = model.coef_.mean(axis=0)
            intercept = model.intercept_.mean()
        else:
            coefs = model.coef_.flatten()
            intercept = float(model.intercept_[0]) if hasattr(model.intercept_, '__len__') else float(model.intercept_)

        # Fisher information for SE estimation
        y_prob = model.predict_proba(X_train)
        if n_classes == 2:
            p_hat = y_prob[:, 1]
            W = p_hat * (1 - p_hat)
            W = np.clip(W, 1e-10, None)
            X_with_int = np.column_stack([np.ones(len(X_train)), X_train])
            XtWX = X_with_int.T @ np.diag(W) @ X_with_int
            try:
                cov_matrix = np.linalg.inv(XtWX)
            except np.linalg.LinAlgError:
                cov_matrix = np.linalg.pinv(XtWX)
        else:
            # Multiclass: approximate with OVR average variance
            X_with_int = np.column_stack([np.ones(len(X_train)), X_train])
            cov_accum = np.zeros((X_with_int.shape[1], X_with_int.shape[1]))
            for c in range(n_classes):
                p_c = y_prob[:, c]
                W_c = p_c * (1 - p_c)
                W_c = np.clip(W_c, 1e-10, None)
                XtWX_c = X_with_int.T @ np.diag(W_c) @ X_with_int
                try:
                    cov_accum += np.linalg.inv(XtWX_c)
                except np.linalg.LinAlgError:
                    cov_accum += np.linalg.pinv(XtWX_c)
            cov_matrix = cov_accum / n_classes

        all_coefs = np.concatenate([[intercept], coefs])
        all_names = ["(Intercepto)"] + list(features)
        se_all = np.sqrt(np.abs(np.diag(cov_matrix)))

        table = []
        for i, name in enumerate(all_names):
            coef = float(all_coefs[i])
            se = float(se_all[i]) if se_all[i] > 0 else 1e-12
            wald = (coef / se) ** 2
            p_val = 1 - sp_stats.chi2.cdf(wald, 1)
            exp_b = math.exp(min(max(coef, -500), 500))
            lower_coef = coef - 1.96 * se
            upper_coef = coef + 1.96 * se
            exp_lower = math.exp(min(max(lower_coef, -500), 500))
            exp_upper = math.exp(min(max(upper_coef, -500), 500))
            table.append({
                "name": name,
                "coeff": _safe(coef), "se": _safe(se),
                "wald": _safe(wald), "p_value": _safe(p_val),
                "exp_b": _safe(exp_b),
                "lower": _safe(exp_lower), "upper": _safe(exp_upper),
                "significant": bool(p_val < 0.05),
            })
        return table
    except Exception:
        return []


def _variable_recommendation(coeff_table):
    """Generate recommendation text for significant variables."""
    if not coeff_table:
        return ""
    sig_vars = [r for r in coeff_table if r["significant"] and r["name"] != "(Intercepto)"]
    nonsig_vars = [r for r in coeff_table if not r["significant"] and r["name"] != "(Intercepto)"]

    if not sig_vars:
        return "Nenhuma variável apresentou significância estatística (p < 0.05). Considere revisar as features selecionadas, aumentar o volume de dados ou verificar a adequação do modelo."

    # Sort by p-value ascending (most significant first)
    sig_vars.sort(key=lambda x: x["p_value"] if x["p_value"] is not None else 1)

    parts = []
    parts.append(f"<strong>{len(sig_vars)}</strong> variável(eis) estatisticamente significativa(s) (p &lt; 0.05):")
    for v in sig_vars:
        p_str = f"{v['p_value']:.10f}".rstrip('0').rstrip('.') if v['p_value'] is not None and v['p_value'] >= 1e-10 else "&lt; 0.0000000001"
        direction = "positivo" if (v['coeff'] or 0) > 0 else "negativo"
        exp_str = f"{v['exp_b']:.10f}".rstrip('0').rstrip('.')
        parts.append(f"&nbsp;&nbsp;→ <strong>{v['name']}</strong> (p = {p_str}, efeito {direction}, Exp(B) = {exp_str})" if v['exp_b'] is not None else f"&nbsp;&nbsp;→ <strong>{v['name']}</strong> (p = {p_str}, efeito {direction})")

    if nonsig_vars:
        names = ", ".join(v["name"] for v in nonsig_vars)
        parts.append(f"Variáveis <strong>não significativas</strong> (p ≥ 0.05): {names}. Considere removê-las para simplificar o modelo.")

    return "<br>".join(parts)


def _run_linear(X, y, features, target, work, target_encoded=False):
    """Linear regression matching Real Statistics (XRealStatsX) output — 100% of observations."""
    X_all = X.values
    y_all = y.values
    n = len(y_all)
    p = X_all.shape[1]  # number of predictors (without intercept)

    model = LinearRegression()
    model.fit(X_all, y_all)
    y_pred = model.predict(X_all)
    residuals = y_all - y_pred

    # ---------- OVERALL FIT (Regression Statistics) ----------
    y_mean = float(np.mean(y_all))
    ss_total = float(np.sum((y_all - y_mean) ** 2))
    ss_residual = float(np.sum(residuals ** 2))
    ss_regression = ss_total - ss_residual

    df_regression = p
    df_residual = n - p - 1
    df_total = n - 1

    ms_regression = ss_regression / df_regression if df_regression > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 1e-15

    f_stat = ms_regression / ms_residual if ms_residual > 0 else 0
    f_p_value = 1 - sp_stats.f.cdf(f_stat, df_regression, df_residual) if df_residual > 0 else 1

    r2 = ss_regression / ss_total if ss_total > 0 else 0
    r2_adj = 1 - (1 - r2) * df_total / df_residual if df_residual > 0 else r2
    multiple_r = math.sqrt(max(r2, 0))
    std_error_reg = math.sqrt(ms_residual)

    # AIC / AICc / SBC (BIC) — Real Statistics formulas
    aic = n * math.log(ss_residual / n) + 2 * (p + 1) if n > 0 and ss_residual > 0 else None
    aicc = aic + 2 * (p + 2) * (p + 3) / (n - p - 3) if aic is not None and (n - p - 3) > 0 else None
    sbc = n * math.log(ss_residual / n) + (p + 1) * math.log(n) if n > 0 and ss_residual > 0 else None

    regression_stats = {
        "multiple_r": _safe(multiple_r),
        "r_square": _safe(r2),
        "r_square_adj": _safe(r2_adj),
        "std_error": _safe(std_error_reg),
        "observations": n,
        "aic": _safe(aic),
        "aicc": _safe(aicc),
        "sbc": _safe(sbc),
    }

    anova = {
        "regression": {
            "df": df_regression, "ss": _safe(ss_regression),
            "ms": _safe(ms_regression), "f": _safe(f_stat),
            "f_significance": _safe(f_p_value),
        },
        "residual": {
            "df": df_residual, "ss": _safe(ss_residual),
            "ms": _safe(ms_residual),
        },
        "total": {
            "df": df_total, "ss": _safe(ss_total),
        },
    }

    # ---------- COEFFICIENT TABLE (coeff, std err, t, p-value, lower, upper, VIF) ----------
    coeff_table = []
    try:
        X_with_int = np.column_stack([np.ones(n), X_all])
        XtX_inv = np.linalg.inv(X_with_int.T @ X_with_int)
        var_covar = XtX_inv * ms_residual
        se_all = np.sqrt(np.diag(var_covar))
        all_coefs = np.concatenate([[model.intercept_], model.coef_])
        all_names = ["(Intercepto)"] + list(features)

        t_crit = sp_stats.t.ppf(0.975, df_residual) if df_residual > 0 else 1.96

        # VIF: 1/(1 - R²_j) where R²_j = regressing x_j on all other x variables
        vif_values = [None]  # intercept has no VIF
        if p > 1:
            for j in range(p):
                others = [k for k in range(p) if k != j]
                X_others = X_all[:, others]
                X_j = X_all[:, j]
                from sklearn.linear_model import LinearRegression as _LR
                _m = _LR().fit(X_others, X_j)
                ss_tot_j = np.sum((X_j - X_j.mean()) ** 2)
                ss_res_j = np.sum((X_j - _m.predict(X_others)) ** 2)
                r2_j = 1 - ss_res_j / ss_tot_j if ss_tot_j > 0 else 0
                vif_values.append(_safe(1 / (1 - r2_j)) if r2_j < 1 else None)
        else:
            vif_values.append(None)

        for i, name in enumerate(all_names):
            coef = float(all_coefs[i])
            se = float(se_all[i]) if se_all[i] > 0 else 1e-12
            t_val = coef / se
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_val), df_residual)) if df_residual > 0 else 1
            lower = coef - t_crit * se
            upper = coef + t_crit * se
            coeff_table.append({
                "name": name,
                "coeff": _safe(coef), "se": _safe(se),
                "wald": _safe(t_val), "p_value": _safe(p_val),
                "exp_b": _safe(math.exp(min(max(coef, -500), 500))),
                "lower": _safe(lower), "upper": _safe(upper),
                "vif": vif_values[i] if i < len(vif_values) else None,
                "significant": bool(p_val < 0.05),
            })
    except Exception:
        coeff_table = _coeff_table_linear(X_all, y_all, model, features)

    recommendation = _variable_recommendation(coeff_table)

    # ---------- RESIDUAL OUTPUT ----------
    std_residuals = residuals / std_error_reg if std_error_reg > 0 else residuals
    max_residuals = min(n, 500)  # cap for performance
    residual_output = [
        {
            "obs": int(i + 1),
            "predicted": _safe(float(y_pred[i])),
            "residual": _safe(float(residuals[i])),
            "std_residual": _safe(float(std_residuals[i])),
        }
        for i in range(max_residuals)
    ]

    # ---------- DURBIN-WATSON ----------
    dw = None
    if n > 1:
        diff_res = np.diff(residuals)
        dw = _safe(float(np.sum(diff_res ** 2) / ss_residual)) if ss_residual > 0 else None

    # ---------- Classification metrics (binarization for compatibility) ----------
    median_val = float(np.median(y_all))
    y_true_bin = (y_all > median_val).astype(int)
    y_pred_bin = (y_pred > median_val).astype(int)
    y_range = max(y_pred.max() - y_pred.min(), 1e-9)
    y_prob_lin = (y_pred - y_pred.min()) / y_range
    clf_metrics = _classification_metrics(y_true_bin, y_pred_bin, y_prob_lin)

    try:
        mape = _safe(mean_absolute_percentage_error(y_all, y_pred))
    except Exception:
        mape = None

    return {
        "model_type": "linear", "target": target, "features": features,
        "regression_stats": regression_stats,
        "anova": anova,
        "metrics": {
            "r2": _safe(r2), "r2_adj": _safe(r2_adj),
            "mae": _safe(mean_absolute_error(y_all, y_pred)),
            "mse": _safe(mean_squared_error(y_all, y_pred)),
            "rmse": _safe(float(np.sqrt(mean_squared_error(y_all, y_pred)))),
            "mape": mape,
            "explained_var": _safe(explained_variance_score(y_all, y_pred)),
        },
        "classification_metrics": clf_metrics,
        "coeff_table": coeff_table,
        "recommendation": recommendation,
        "coefficients": {f: _safe(c) for f, c in zip(features, model.coef_)},
        "intercept": _safe(model.intercept_),
        "actual": [_safe(v) for v in y_all[:500]],
        "predicted": [_safe(v) for v in y_pred[:500]],
        "residual_output": residual_output,
        "durbin_watson": dw,
        "observations": n,
    }


def _run_logistic(X, y, features, target, work):
    """Logistic regression matching Real Statistics (XRealStatsX) output — 100% of observations."""
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y.astype(str))
    class_names = le_target.classes_.tolist()
    n_classes = len(class_names)
    is_binary = n_classes == 2

    X_all = X.values
    y_all = y_enc
    n = len(y_all)
    p = X_all.shape[1]

    # Fit on ALL data
    try:
        model = LogisticRegression(max_iter=5000, random_state=42, solver="lbfgs")
        model.fit(X_all, y_all)
    except Exception:
        try:
            scaler = StandardScaler()
            X_all = scaler.fit_transform(X_all)
            model = LogisticRegression(max_iter=5000, random_state=42, solver="saga")
            model.fit(X_all, y_all)
        except Exception as e:
            return {"error": f"Erro ao treinar modelo logístico: {str(e)[:200]}"}

    y_pred = model.predict(X_all)

    # Probabilities
    try:
        y_prob = model.predict_proba(X_all)
    except Exception:
        y_prob = None

    cm = confusion_matrix(y_all, y_pred)
    clf_metrics = _classification_metrics(y_all, y_pred, y_prob)

    # ---------- SIGNIFICANCE TESTING & R-SQUARE (Real Statistics structure) ----------
    model_summary = None
    omnibus_test = None
    try:
        eps = 1e-15
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)

        # LL1 (fitted model)
        ll_model = sum(math.log(y_prob_clipped[i, y_all[i]]) for i in range(n))

        # LL0 (null model — class frequencies only)
        class_freq = np.bincount(y_all, minlength=n_classes) / n
        class_freq = np.clip(class_freq, eps, 1 - eps)
        ll_null = sum(math.log(class_freq[y_all[i]]) for i in range(n))

        # Chi-square (Omnibus / Likelihood Ratio Test)
        chi2_model = 2 * (ll_model - ll_null)
        df_model = p
        chi2_p_value = 1 - sp_stats.chi2.cdf(chi2_model, df_model) if df_model > 0 else 1

        # Pseudo R-squared (Real Statistics: L=McFadden, CS=Cox&Snell, N=Nagelkerke)
        mcfadden = 1 - (ll_model / ll_null) if ll_null != 0 else 0
        cox_snell = 1 - math.exp((-2 / n) * (ll_model - ll_null))
        cox_snell_max = 1 - math.exp((2 * ll_null) / n)
        nagelkerke = cox_snell / cox_snell_max if cox_snell_max > 0 else 0

        # AIC / BIC (Real Statistics formulas)
        aic = -2 * ll_model + 2 * (p + 1)
        bic = -2 * ll_model + (p + 1) * math.log(n)

        model_summary = {
            "ll0": _safe(ll_null),
            "ll1": _safe(ll_model),
            "neg2ll": _safe(-2 * ll_model),
            "neg2ll_null": _safe(-2 * ll_null),
            "mcfadden_r2": _safe(mcfadden),
            "cox_snell_r2": _safe(cox_snell),
            "nagelkerke_r2": _safe(nagelkerke),
            "aic": _safe(aic),
            "bic": _safe(bic),
            "observations": n,
        }

        omnibus_test = {
            "chi2": _safe(chi2_model),
            "df": df_model,
            "p_value": _safe(chi2_p_value),
        }
    except Exception:
        pass

    # ---------- COEFFICIENT TABLE (coeff b, s.e., Wald, p-value, exp(b), lower, upper) ----------
    coeff_table = _coeff_table_logistic(X_all, y_all, model, features)
    recommendation = _variable_recommendation(coeff_table)

    # ---------- CLASSIFICATION TABLE ----------
    class_accuracy = []
    for i, cn in enumerate(class_names):
        total_actual = int(cm[i].sum())
        correct = int(cm[i, i])
        pct = _safe(correct / total_actual * 100) if total_actual > 0 else 0
        class_accuracy.append({"class": cn, "total": total_actual, "correct": correct, "pct": pct})
    overall_correct = int(np.trace(cm))
    overall_pct = _safe(overall_correct / n * 100) if n > 0 else 0

    # ROC curve data (binary only)
    roc_curve_data = None
    if is_binary and y_prob is not None:
        try:
            from sklearn.metrics import roc_curve
            prob_pos = y_prob[:, 1]
            fpr, tpr, _ = roc_curve(y_all, prob_pos)
            step = max(1, len(fpr) // 100)
            roc_curve_data = {
                "fpr": [_safe(v) for v in fpr[::step]],
                "tpr": [_safe(v) for v in tpr[::step]],
            }
        except Exception:
            pass

    return {
        "model_type": "logistic", "target": target, "features": features,
        "model_summary": model_summary,
        "omnibus_test": omnibus_test,
        "classification_metrics": clf_metrics,
        "coeff_table": coeff_table,
        "recommendation": recommendation,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "class_accuracy": class_accuracy,
        "overall_accuracy": overall_pct,
        "roc_curve": roc_curve_data,
        "observations": n,
    }



def _run_clustering(df, features, n_clusters: int = 0):
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

    # Evaluate k range (2-10)
    max_k = min(10, len(X) // 3)
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

    # Determine best k
    auto_k = True
    if n_clusters and n_clusters >= 2:
        best_k = min(n_clusters, max_k)
        auto_k = False
    else:
        best_k = max(silhouettes, key=lambda s: s["silhouette"] or 0)["k"]

    # Build rationale
    if auto_k:
        best_sil = next(s for s in silhouettes if s["k"] == best_k)
        rationale = (
            f"K={best_k} selecionado automaticamente. "
            f"Critério: maior Silhouette Score ({best_sil['silhouette']:.4f}). "
        )
        # Detect elbow (largest drop in inertia)
        if len(inertias) >= 3:
            drops = []
            for i in range(1, len(inertias)):
                prev = inertias[i - 1]["inertia"]
                curr = inertias[i]["inertia"]
                drop_pct = (prev - curr) / prev * 100 if prev > 0 else 0
                drops.append({"k": inertias[i]["k"], "drop_pct": round(drop_pct, 1)})
            # Elbow = where drop % decreases most sharply
            max_drop_k = max(drops, key=lambda d: d["drop_pct"])["k"]
            rationale += (
                f"Pelo método do cotovelo, a maior queda percentual de inertia ocorre em K={max_drop_k}. "
                f"A partir desse ponto, acrescentar clusters gera ganho marginal decrescente."
            )
        else:
            rationale += "Dados insuficientes para análise de cotovelo detalhada."
    else:
        rationale = f"K={best_k} definido manualmente pelo usuário."

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = km_final.fit_predict(X_scaled)
    final_sil = _safe(silhouette_score(X_scaled, final_labels))
    centroids = km_final.cluster_centers_

    # Clustering quality metrics
    try:
        calinski = _safe(calinski_harabasz_score(X_scaled, final_labels))
    except Exception:
        calinski = None
    try:
        davies = _safe(davies_bouldin_score(X_scaled, final_labels))
    except Exception:
        davies = None

    # Euclidean distance matrix between centroids
    from scipy.spatial.distance import cdist
    euclidean_matrix = cdist(centroids, centroids, metric="euclidean")
    euclidean_data = {
        "labels": [f"C{i}" for i in range(best_k)],
        "values": [[_safe(euclidean_matrix[i][j]) for j in range(best_k)] for i in range(best_k)],
    }

    # Cluster sizes
    unique, counts = np.unique(final_labels, return_counts=True)
    cluster_sizes = [{"cluster": int(u), "size": int(c)} for u, c in zip(unique, counts)]

    # Cluster profiles (original scale)
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

    # Scatter (first 2 features)
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
        "best_k": best_k, "auto_k": auto_k, "rationale": rationale,
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
        "euclidean": euclidean_data,
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

        /* Coefficient statistics table */
        .aa-coeff-stats-table th {{ white-space:nowrap; }}
        .aa-coeff-stats-table td {{ font-family:'JetBrains Mono',monospace; font-size:11px; }}
        .aa-row-significant {{ background:rgba(57,211,83,0.05) !important; }}
        .aa-row-significant:hover {{ background:rgba(57,211,83,0.1) !important; }}

        /* Tooltips */
        .aa-tooltip-trigger {{ position:relative; cursor:help; white-space:nowrap; }}
        .aa-tooltip-icon {{ display:inline-flex; align-items:center; justify-content:center; width:14px; height:14px; border-radius:50%; background:#30363d; color:#8b949e; font-size:9px; font-weight:700; margin-left:3px; vertical-align:middle; }}
        .aa-tooltip-text {{ visibility:hidden; opacity:0; position:absolute; bottom:calc(100% + 8px); left:50%; transform:translateX(-50%); background:#1c2128; border:1px solid #444c56; border-radius:8px; padding:10px 12px; font-size:11px; font-weight:400; color:#c9d1d9; line-height:1.5; width:280px; white-space:normal; z-index:100; pointer-events:none; box-shadow:0 4px 12px rgba(0,0,0,0.4); transition:opacity 0.2s; }}
        .aa-tooltip-text::after {{ content:''; position:absolute; top:100%; left:50%; transform:translateX(-50%); border:6px solid transparent; border-top-color:#444c56; }}
        .aa-tooltip-trigger:hover .aa-tooltip-text {{ visibility:visible; opacity:1; }}

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

            <div class="aa-form-group" id="predClustersGroup" style="display:none;">
                <label class="aa-label">Quantidade de Clusters (0 = automático)</label>
                <input id="predNClusters" type="number" min="0" max="20" value="0"
                       class="aa-select" style="font-family:'JetBrains Mono',monospace;">
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
    const clustersGroup = document.getElementById('predClustersGroup');
    if (mt === 'linear') {{
        info.textContent = 'Regressão Linear: prevê um valor numérico contínuo. Variáveis categóricas serão codificadas automaticamente (Label Encoding).';
        targetGroup.style.display = 'block';
        clustersGroup.style.display = 'none';
    }} else if (mt === 'logistic') {{
        info.textContent = 'Regressão Logística: classifica em categorias. Métricas: AUC, KS, Precision, Recall, F1, Acurácia + Matriz de Confusão.';
        targetGroup.style.display = 'block';
        clustersGroup.style.display = 'none';
    }} else {{
        info.textContent = 'Clusterização K-Means: agrupa dados similares. Informe a quantidade de clusters ou deixe 0 para detecção automática via Silhouette Score.';
        targetGroup.style.display = 'none';
        clustersGroup.style.display = 'block';
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
        const body = {{ query_data: DATA, target: modelType === 'clustering' ? '' : target, features, model_type: modelType, n_clusters: modelType === 'clustering' ? parseInt(document.getElementById('predNClusters').value) || 0 : 0 }};
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

function renderCoeffTable(table, recommendation, modelType) {{
    if (!table || table.length === 0) return '';
    const isLogistic = modelType === 'logistic';
    const hasVIF = !isLogistic && table.some(r => r.vif !== null && r.vif !== undefined);
    const tooltips = {{
        coeff: 'Coeficiente (B): magnitude e direção do efeito da variável sobre o alvo. Positivo = aumenta; Negativo = diminui.',
        se: 'Erro Padrão (S.E.): incerteza da estimativa do coeficiente. Quanto menor, mais precisa a estimativa.',
        wald: isLogistic
            ? 'Wald (χ²): teste de significância — (B / S.E.)². Valores altos indicam que o coeficiente é significativamente diferente de zero.'
            : 'Estatística t: teste de significância — B / S.E. Valores com |t| > 2 geralmente indicam significância.',
        p_value: 'p-valor: probabilidade de observar este efeito por acaso. p < 0.05 = estatisticamente significativo (95% de confiança).',
        exp_b: isLogistic
            ? 'Exp(B) — Odds Ratio: multiplicador da chance. >1 = aumenta a chance; <1 = diminui a chance; =1 = sem efeito.'
            : 'Exp(B): exponencial do coeficiente.',
        lower: isLogistic
            ? 'Limite Inferior do IC 95% para Exp(B). Se o intervalo contém 1, o efeito pode não ser significativo.'
            : 'Limite Inferior do IC 95% para o coeficiente. Se o intervalo contém 0, o efeito pode não ser significativo.',
        upper: isLogistic
            ? 'Limite Superior do IC 95% para Exp(B).'
            : 'Limite Superior do IC 95% para o coeficiente.',
        vif: 'VIF (Variance Inflation Factor): detecta multicolinearidade. VIF > 5 indica correlação alta entre preditores; VIF > 10 é problemático.',
    }};

    const th = (label, key) => `<th><span class="aa-tooltip-trigger">${{label}} <span class="aa-tooltip-icon">?</span><span class="aa-tooltip-text">${{tooltips[key]}}</span></span></th>`;

    let html = `<div class="aa-section"><div class="aa-section-title">Tabela de Coeficientes</div>
    <div class="aa-card" style="overflow-x:auto;">
    <table class="aa-freq-table aa-coeff-stats-table">
    <thead><tr>
        <th>Variável</th>
        ${{th('coeff', 'coeff')}}
        ${{th('std err', 'se')}}
        ${{th(isLogistic ? 'Wald' : 't stat', 'wald')}}
        ${{th('p-value', 'p_value')}}
        ${{isLogistic ? th('exp(b)', 'exp_b') : ''}}
        ${{th('lower', 'lower')}}
        ${{th('upper', 'upper')}}
        ${{hasVIF ? th('vif', 'vif') : ''}}
    </tr></thead>
    <tbody>`;

    table.forEach(row => {{
        const sig = row.significant;
        const rowClass = sig ? 'aa-row-significant' : '';
        const pFmt = row.p_value !== null ? (row.p_value < 0.0000000001 ? '< 0.0000000001' : fmtC(row.p_value)) : '—';
        const pColor = sig ? '#39d353' : (row.p_value !== null && row.p_value < 0.1 ? '#f0883e' : '#8b949e');
        const nameStyle = row.name === '(Intercepto)' ? 'color:#8b949e;font-style:italic;' : (sig ? 'color:#58a6ff;font-weight:600;' : '');

        html += `<tr class="${{rowClass}}">
            <td style="${{nameStyle}}">${{row.name}} ${{sig ? '<span style="color:#39d353;font-size:10px;">★</span>' : ''}}</td>
            <td>${{fmtC(row.coeff)}}</td>
            <td>${{fmtC(row.se)}}</td>
            <td>${{fmtC(row.wald)}}</td>
            <td style="color:${{pColor}};font-weight:${{sig ? '600' : 'normal'}}">${{pFmt}}</td>
            ${{isLogistic ? `<td>${{fmtC(row.exp_b)}}</td>` : ''}}
            <td>${{fmtC(row.lower)}}</td>
            <td>${{fmtC(row.upper)}}</td>
            ${{hasVIF ? `<td style="color:${{row.vif && row.vif > 5 ? '#f0883e' : '#8b949e'}}">${{row.vif !== null && row.vif !== undefined ? fmtC(row.vif) : ''}}</td>` : ''}}
        </tr>`;
    }});

    html += `</tbody></table>
    <div style="margin-top:8px;font-size:10px;color:#8b949e;">
        ★ = significativo (p &lt; 0.05) &nbsp;|&nbsp; IC = Intervalo de Confiança 95%${{isLogistic ? ' para Exp(B)' : ''}}
    </div></div></div>`;

    // Recommendation
    if (recommendation) {{
        html += `<div class="aa-section"><div class="aa-section-title">Recomendação de Variáveis</div>
        <div class="aa-card" style="border-left:3px solid #58a6ff;padding:14px 16px;">
            <div style="font-size:12px;line-height:1.7;color:#c9d1d9;">${{recommendation}}</div>
        </div></div>`;
    }}

    return html;
}}

function renderLinearResult(r) {{
    const m = r.metrics;
    const rs = r.regression_stats;
    const an = r.anova;

    let html = `<div class="aa-section"><div class="aa-section-title">Regressão Linear — ${{r.target}}</div>`;

    // --- OVERALL FIT (Real Statistics) ---
    if (rs) {{
        html += `<div class="aa-grid" style="grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
        <div class="aa-card">
        <div class="aa-card-title">OVERALL FIT</div>
        <table class="aa-freq-table" style="max-width:100%;">
            <tbody>
                <tr><td style="color:#8b949e;">Multiple R</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(rs.multiple_r)}}</td></tr>
                <tr><td style="color:#8b949e;">R Square</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(rs.r_square)}}</td></tr>
                <tr><td style="color:#8b949e;">Adjusted R Square</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(rs.r_square_adj)}}</td></tr>
                <tr><td style="color:#8b949e;">Standard Error</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(rs.std_error)}}</td></tr>
                <tr><td style="color:#8b949e;">Observations</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{rs.observations}}</td></tr>
            </tbody>
        </table></div>
        <div class="aa-card">
        <div class="aa-card-title">&nbsp;</div>
        <table class="aa-freq-table" style="max-width:100%;">
            <tbody>
                <tr><td style="color:#8b949e;">AIC</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{rs.aic !== null ? fmtC(rs.aic) : '—'}}</td></tr>
                <tr><td style="color:#8b949e;">AICc</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{rs.aicc !== null ? fmtC(rs.aicc) : '—'}}</td></tr>
                <tr><td style="color:#8b949e;">SBC (BIC)</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{rs.sbc !== null ? fmtC(rs.sbc) : '—'}}</td></tr>
                ${{r.durbin_watson !== null && r.durbin_watson !== undefined ? `<tr><td style="color:#8b949e;">Durbin-Watson</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(r.durbin_watson)}}</td></tr>` : ''}}
            </tbody>
        </table></div></div>`;
    }}

    // --- ANOVA ---
    if (an) {{
        const pFmt = an.regression.f_significance !== null
            ? (an.regression.f_significance < 0.0000000001 ? '< 0.0000000001' : fmtC(an.regression.f_significance))
            : '—';
        const pColor = an.regression.f_significance !== null && an.regression.f_significance < 0.05 ? '#39d353' : '#8b949e';
        const sig = an.regression.f_significance !== null && an.regression.f_significance < 0.05 ? 'yes' : 'no';
        const sigColor = sig === 'yes' ? '#39d353' : '#8b949e';
        html += `<div class="aa-card" style="margin-bottom:16px;">
        <div class="aa-card-title">ANOVA</div>
        <div style="overflow-x:auto;">
        <table class="aa-freq-table aa-coeff-stats-table">
            <thead><tr>
                <th></th><th>df</th><th>SS</th><th>MS</th><th>F</th><th>p-value</th><th>sig</th>
            </tr></thead>
            <tbody>
                <tr>
                    <td style="font-weight:600;color:#58a6ff;">Regression</td>
                    <td>${{an.regression.df}}</td>
                    <td>${{fmtC(an.regression.ss)}}</td>
                    <td>${{fmtC(an.regression.ms)}}</td>
                    <td style="font-weight:600;">${{fmtC(an.regression.f)}}</td>
                    <td style="color:${{pColor}};font-weight:600;">${{pFmt}}</td>
                    <td style="color:${{sigColor}};font-weight:600;text-align:center;">${{sig}}</td>
                </tr>
                <tr>
                    <td style="font-weight:600;color:#58a6ff;">Residual</td>
                    <td>${{an.residual.df}}</td>
                    <td>${{fmtC(an.residual.ss)}}</td>
                    <td>${{fmtC(an.residual.ms)}}</td>
                    <td></td><td></td><td></td>
                </tr>
                <tr style="border-top:1px solid #30363d;">
                    <td style="font-weight:600;color:#58a6ff;">Total</td>
                    <td>${{an.total.df}}</td>
                    <td>${{fmtC(an.total.ss)}}</td>
                    <td></td><td></td><td></td><td></td>
                </tr>
            </tbody>
        </table></div></div>`;
    }}

    html += `</div>`;

    // Coefficient table (with VIF)
    html += renderCoeffTable(r.coeff_table, r.recommendation, 'linear');

    // Metrics cards
    html += `<div class="aa-section"><div class="aa-section-title">Métricas (100% das observações)</div>
    <div class="aa-grid aa-grid-6" style="margin-bottom:16px;">
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.mae)}}</div><div class="aa-metric-label">MAE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.mse)}}</div><div class="aa-metric-label">MSE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.rmse)}}</div><div class="aa-metric-label">RMSE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{m.mape !== null ? (m.mape * 100).toFixed(1) + '%' : '—'}}</div><div class="aa-metric-label">MAPE</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{fmt(m.explained_var)}}</div><div class="aa-metric-label">Var. Explicada</div></div>
        <div class="aa-metric-card"><div class="aa-metric-value">${{r.observations}}</div><div class="aa-metric-label">Observações</div></div>
    </div></div>`;

    html += renderClfMetrics(r.classification_metrics, 'Métricas de Classificação (binarizado pela mediana)');

    html += `<div class="aa-section"><div class="aa-section-title">Real vs Previsto</div><div class="aa-card"><div style="height:300px;"><canvas id="predChart1"></canvas></div></div></div>`;

    // Residual Output (Real Statistics)
    if (r.residual_output && r.residual_output.length > 0) {{
        html += `<div class="aa-section"><div class="aa-section-title">Saída de Resíduos</div>
        <div class="aa-card" style="overflow-x:auto;max-height:400px;overflow-y:auto;">
        <table class="aa-freq-table aa-coeff-stats-table">
        <thead><tr><th>Obs</th><th>Previsto</th><th>Resíduo</th><th>Resíduo Padrão</th></tr></thead>
        <tbody>`;
        r.residual_output.forEach(row => {{
            const absStd = Math.abs(row.std_residual || 0);
            const outlier = absStd > 2 ? 'color:#f0883e;font-weight:600;' : '';
            html += `<tr>
                <td>${{row.obs}}</td>
                <td>${{fmtC(row.predicted)}}</td>
                <td>${{fmtC(row.residual)}}</td>
                <td style="${{outlier}}">${{fmtC(row.std_residual)}}</td>
            </tr>`;
        }});
        html += `</tbody></table></div></div>`;
    }}

    return html;
}}

function renderLogisticResult(r) {{
    let html = `<div class="aa-section"><div class="aa-section-title">Regressão Logística — ${{r.target}}</div>`;

    // --- Significance Testing & R-Square (Real Statistics layout) ---
    const ms = r.model_summary;
    const ot = r.omnibus_test;
    if (ms || ot) {{
        html += `<div class="aa-grid" style="grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">`;

        if (ot && ms) {{
            const pFmt = ot.p_value !== null ? (ot.p_value < 0.0000000001 ? '< 0.0000000001' : fmtC(ot.p_value)) : '—';
            const pColor = ot.p_value !== null && ot.p_value < 0.05 ? '#39d353' : '#8b949e';
            const sig = ot.p_value !== null && ot.p_value < 0.05 ? 'yes' : 'no';
            const sigColor = sig === 'yes' ? '#39d353' : '#8b949e';
            html += `<div class="aa-card">
            <div class="aa-card-title">Significance Testing</div>
            <table class="aa-freq-table" style="max-width:100%;">
                <tbody>
                    <tr><td style="color:#8b949e;">LL0</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(ms.ll0)}}</td></tr>
                    <tr><td style="color:#8b949e;">LL1</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(ms.ll1)}}</td></tr>
                    <tr style="border-top:1px solid #30363d;"><td style="color:#8b949e;">Chi-Sq</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;font-weight:600;">${{fmtC(ot.chi2)}}</td></tr>
                    <tr><td style="color:#8b949e;">df</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{ot.df}}</td></tr>
                    <tr><td style="color:#8b949e;">p-value</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;color:${{pColor}};font-weight:600;">${{pFmt}}</td></tr>
                    <tr><td style="color:#8b949e;">sig</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;color:${{sigColor}};font-weight:600;">${{sig}}</td></tr>
                </tbody>
            </table></div>`;

            html += `<div class="aa-card">
            <div class="aa-card-title">R-Square & Information Criteria</div>
            <table class="aa-freq-table" style="max-width:100%;">
                <tbody>
                    <tr><td style="color:#8b949e;">R-Sq (L) McFadden</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(ms.mcfadden_r2)}}</td></tr>
                    <tr><td style="color:#8b949e;">R-Sq (CS) Cox &amp; Snell</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(ms.cox_snell_r2)}}</td></tr>
                    <tr><td style="color:#8b949e;">R-Sq (N) Nagelkerke</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{fmtC(ms.nagelkerke_r2)}}</td></tr>
                    <tr style="border-top:1px solid #30363d;"><td style="color:#8b949e;">AIC</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{ms.aic !== null && ms.aic !== undefined ? fmtC(ms.aic) : '—'}}</td></tr>
                    <tr><td style="color:#8b949e;">BIC</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{ms.bic !== null && ms.bic !== undefined ? fmtC(ms.bic) : '—'}}</td></tr>
                    <tr><td style="color:#8b949e;">Observations</td><td style="text-align:right;font-family:'JetBrains Mono',monospace;">${{ms.observations}}</td></tr>
                </tbody>
            </table></div>`;
        }}
        html += `</div>`;
    }}

    html += `</div>`;

    // Coefficient table (coeff, s.e., Wald, p-value, exp(b), lower, upper)
    html += renderCoeffTable(r.coeff_table, r.recommendation, 'logistic');

    html += renderClfMetrics(r.classification_metrics, 'Métricas Estatísticas (100% das observações)');

    // Classification Table with accuracy % (Real Statistics style)
    html += `<div class="aa-section"><div class="aa-section-title">Classification Table & Curva ROC</div>
    <div class="aa-grid" style="grid-template-columns:1fr 1fr;gap:16px;">`;

    html += '<div class="aa-card"><div class="aa-card-title">Classification Table</div>';
    if (r.class_names && r.confusion_matrix) {{
        html += '<div style="overflow-x:auto;max-height:400px;overflow-y:auto;"><table class="aa-freq-table"><thead><tr><th style="min-width:80px;">Real \\ Pred</th>';
        r.class_names.forEach(c => html += `<th>${{c}}</th>`);
        html += '<th>% Correct</th></tr></thead><tbody>';
        r.confusion_matrix.forEach((row, i) => {{
            html += `<tr><td style="font-weight:600;color:#58a6ff">${{r.class_names[i]}}</td>`;
            row.forEach((v, j) => {{
                const bg = i === j ? 'rgba(57,211,83,0.15)' : (v > 0 ? 'rgba(255,99,71,0.1)' : '');
                html += `<td style="background:${{bg}}">${{v}}</td>`;
            }});
            const ca = r.class_accuracy ? r.class_accuracy[i] : null;
            const pct = ca ? ca.pct : '—';
            html += `<td style="font-weight:600;color:#39d353;">${{typeof pct === 'number' ? pct.toFixed(1) + '%' : pct}}</td></tr>`;
        }});
        html += `<tr style="border-top:1px solid #30363d;"><td style="font-weight:600;color:#58a6ff;">Overall</td>`;
        r.class_names.forEach(() => html += '<td></td>');
        html += `<td style="font-weight:700;color:#39d353;">${{typeof r.overall_accuracy === 'number' ? r.overall_accuracy.toFixed(1) + '%' : '—'}}</td></tr>`;
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
    let html = `<div class="aa-section"><div class="aa-section-title">Clusterização K-Means — ${{r.best_k}} Clusters ${{r.auto_k ? '(automático)' : '(definido pelo usuário)'}}</div>
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

    // Charts: Scatter + Elbow + Euclidean
    html += `<div class="aa-section"><div class="aa-section-title">Visualizações</div>
    <div class="aa-grid aa-grid-2">
        <div class="aa-card"><div class="aa-card-title">Dispersão (${{r.scatter.x_col}} × ${{r.scatter.y_col}})</div><div style="height:300px;"><canvas id="predChart1"></canvas></div></div>
        <div class="aa-card"><div class="aa-card-title">Método do Cotovelo (Elbow)</div><div style="height:300px;"><canvas id="predChart2"></canvas></div>
            <div class="aa-info" style="margin-top:10px;">${{r.rationale || ''}}</div>
        </div>
    </div>
    <div class="aa-grid aa-grid-2" style="margin-top:16px;">
        <div class="aa-card"><div class="aa-card-title">Distância Euclidiana entre Centróides</div><div style="height:300px;"><canvas id="predChart3"></canvas></div></div>
        <div class="aa-card"><div class="aa-card-title">Matriz de Distância Euclidiana</div>`;

    // Euclidean heatmap table
    if (r.euclidean) {{
        const euc = r.euclidean;
        const n = euc.labels.length;
        html += `<div style="overflow-x:auto;"><div class="corr-grid" style="grid-template-columns:56px repeat(${{n}}, 56px);display:inline-grid;gap:1px;background:#21262d;border-radius:8px;overflow:hidden;">`;
        html += '<div class="corr-cell corr-header"></div>';
        euc.labels.forEach(l => html += `<div class="corr-cell corr-header">${{l}}</div>`);
        const maxDist = Math.max(...euc.values.flat().filter(v => v !== null), 0.01);
        for (let i = 0; i < n; i++) {{
            html += `<div class="corr-cell corr-header" style="width:56px;justify-content:flex-end;padding-right:4px;">${{euc.labels[i]}}</div>`;
            for (let j = 0; j < n; j++) {{
                const v = euc.values[i][j];
                const intensity = i === j ? 0 : (v / maxDist);
                const bg = i === j ? '#161b22' : `rgba(255, 99, 71, ${{(intensity * 0.7 + 0.1).toFixed(2)}})`;
                const tc = intensity > 0.4 ? '#fff' : '#c9d1d9';
                html += `<div class="corr-cell" style="background:${{bg}};color:${{tc}}" title="${{euc.labels[i]}} ↔ ${{euc.labels[j]}}: ${{v !== null ? v.toFixed(3) : '—'}}">${{i === j ? '0' : (v !== null ? v.toFixed(2) : '—')}}</div>`;
            }}
        }}
        html += '</div></div>';
    }}
    html += '</div></div></div>';

    // Cluster profiles
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

        // Euclidean distance bar chart
        const c3 = document.getElementById('predChart3');
        if (c3 && r.euclidean) {{
            const euc = r.euclidean;
            const labels = [];
            const values = [];
            const colors = [];
            for (let i = 0; i < euc.labels.length; i++) {{
                for (let j = i + 1; j < euc.labels.length; j++) {{
                    labels.push(`${{euc.labels[i]}} ↔ ${{euc.labels[j]}}`);
                    values.push(euc.values[i][j]);
                    colors.push(clPalette[(i + j) % clPalette.length] + '99');
                }}
            }}
            new Chart(c3, {{
                type: 'bar',
                data: {{ labels, datasets: [{{ label: 'Distância Euclidiana', data: values, backgroundColor: colors, borderWidth: 0, borderRadius: 4 }}] }},
                options: {{ responsive:true, maintainAspectRatio:false, indexAxis:'y', plugins:{{ legend:{{ display:false }} }}, scales:{{ x:{{ title:{{ display:true, text:'Distância Euclidiana', color:'#c9d1d9', font:{{ size:10 }} }}, beginAtZero:true }}, y:{{ ticks:{{ font:{{ size:10 }} }} }} }} }},
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

function fmtC(v) {{
    if (v === null || v === undefined) return '—';
    if (typeof v !== 'number') return String(v);
    const s = v.toFixed(10);
    return s.replace(/0+$/, '').replace(/\.$/, '.0');
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

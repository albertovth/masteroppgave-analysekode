# -*- coding: utf-8 -*-

# Konfigurerer miljøet: importerer pakker, peker mot syntetiske Excel-filer og stopper realdata.
# Input er stioppsett/kolonnenavn, output er Path-objekter (xlsx_path/pbg_path) og profiler til senere steg.
# =======================
# STEG 0: Importer + konfig
# =======================
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Optional, Union
import re
import sys
import atexit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dirichlet import dirichlet as _dir_pkg
from scipy.optimize import minimize_scalar
from scipy.special import gammaln  # robust LL-formel
from scipy.stats import chi2, f_oneway, mannwhitneyu, norm, spearmanr
import statsmodels.api as sm

from export_helpers import export_excel
from model_metrics import compute_aic, compute_bic, compute_pseudo_r2

_chi2 = chi2

# --- Output (modul, metadata) ---
_output_manifest = []

def register_output(path, kind, df=None, note="", step="", label=""):
    try:
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import numpy as _np
        _path = str(path) if path is not None else ""
        _exists = False
        _size = _np.nan
        if _path:
            try:
                _p = _Path(_path)
                _exists = _p.exists()
                if _exists:
                    _size = _p.stat().st_size
            except Exception:
                pass
        _shape = ""
        _cols = ""
        if df is not None:
            try:
                _shape = str(getattr(df, "shape", ""))
                cols = getattr(df, "columns", None)
                if cols is not None:
                    _cols = ",".join([str(c) for c in list(cols)])
            except Exception:
                pass
        _output_manifest.append({
            "timestamp_iso": _dt.now().isoformat(timespec="seconds"),
            "step": step,
            "label": label,
            "path": _path,
            "kind": kind,
            "exists": _exists,
            "size_bytes": _size,
            "shape": _shape,
            "columns": _cols,
            "note": note,
        })
    except Exception:
        pass

def write_output_manifest(data_dir, data_file_stem, run_mode):
    try:
        from datetime import datetime as _dt
        from pathlib import Path as _Path
        import pandas as _pd
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        stem = data_file_stem or "run"
        base = _Path(data_dir) / f"{stem}_outputs_manifest_{run_mode}_{ts}"
        df = _pd.DataFrame(_output_manifest)
        out_xlsx = base.with_suffix(".xlsx")
        out_csv = base.with_suffix(".csv")
        try:
            with _pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as w:
                df.to_excel(w, index=False, sheet_name="manifest")
        except Exception:
            with _pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
                df.to_excel(w, index=False, sheet_name="manifest")
        try:
            df.to_csv(out_csv, index=False, encoding="utf-8")
        except Exception:
            pass
        return out_xlsx, out_csv, df
    except Exception:
        return None, None, None

def run_all_steps():
    
    
    
    # Trygge defaults for samle-tabeller som ikke alltid settes
    BETA_JOINT = pd.DataFrame()
    WALD_JOINT = pd.DataFrame()
    WALD_BLOCKRED = pd.DataFrame()
    models_S: dict = {}
    models_O: dict = {}
    X_all = pd.DataFrame()
    X_rev = pd.DataFrame()
    Z_S = pd.DataFrame()
    Z_O = pd.DataFrame()
    DIR_RES = pd.DataFrame()

    # HC3 bruk (logg)
    _hc3_inventory = [
        "STEG 10 _fit_variant (OLS robust cov)",
        "STEG 10 coef tables",
        "STEG 12 _coef_table_direct",
        "STEG 12 _wald_joint_table",
        "STEG 12A ILR-only (coef + wald)",
        "STEG 12B CLR-equivalent (cov propagation)",
        "STEG 12D block reduction",
    ]
    print("[HC3] Robust covariance sites:", "; ".join(_hc3_inventory))
    
    REDACT_LOGS = os.getenv("REDACT_LOGS", "0").strip().lower() in ("1", "true", "yes", "y")
    print(f"[RUN_META] REDACT_LOGS={int(REDACT_LOGS)}")

    def export_excel(
        df: pd.DataFrame,
        *,
        path: Union[str, Path, None] = None,
        writer: Optional[pd.ExcelWriter] = None,
        sheet_name: Optional[str] = None,
        label: str = "",
    ) -> None:
        if writer is None and path is None:
            raise ValueError("Må enten gi path, eller writer + sheet_name")
        if writer is not None and sheet_name is None:
            raise ValueError("Må oppgi sheet_name når writer brukes")
        file_path: Optional[Path] = None
        if writer is None:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(file_path, index=False)
        else:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            file_path = Path(getattr(writer, "path", "")) if getattr(writer, "path", None) else None
        if not label:
            label = sheet_name or (file_path.stem if file_path is not None else "")
        print("[export_excel]" + (f" {label}" if label else ""))
        if file_path is not None:
            print(f"  Fil: {file_path}")
        elif writer is not None:
            maybe = getattr(writer, "path", None)
            if maybe:
                print(f"  Fil: {maybe}")
        print(f"  Shape: {df.shape}")
        print(f"  Kolonner: {list(df.columns)}")
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False).iloc[:15]
        if not miss.empty:
            miss_txt = "; ".join(f"{k}={int(v)}" for k, v in miss.items())
            print(f"  Missing (top 15): {miss_txt}")

    def _log_df_meta(label, df):
        print(label)
        if df is None or getattr(df, "empty", False):
            print("  (ingen data)")
            return
        print(f"  Shape: {df.shape}")
        print(f"  Kolonner: {list(df.columns)}")
        miss = df.isna().sum()
        miss = miss[miss > 0].sort_values(ascending=False).iloc[:15]
        if not miss.empty:
            miss_txt = "; ".join(f"{k}={int(v)}" for k, v in miss.items())
            print(f"  Missing (top 15): {miss_txt}")

    def _log_counts_summary(label, counts_series):
        print(label)
        if counts_series is None or getattr(counts_series, "empty", False):
            print("  (ingen data)")
            return
        vals = pd.to_numeric(counts_series, errors="coerce").dropna()
        if vals.empty:
            print("  (ingen numeriske data)")
            return
        print(
            f"  n={len(vals)}, min={vals.min()}, "
            f"median={vals.median()}, max={vals.max()}, mean={vals.mean():.3f}"
        )

    def _log_small_cats(label, df, max_uniques=30, max_rows=30):
        if df is None or getattr(df, "empty", False):
            return
        for col in df.columns:
            if col == "ID":
                continue
            s = df[col]
            if str(s.dtype).startswith("object") or str(s.dtype).startswith("category"):
                uniq = s.nunique(dropna=True)
                if uniq <= max_uniques:
                    vc = s.value_counts(dropna=False).iloc[:max_rows]
                    vc_txt = "; ".join(f"{k}={v}" for k, v in vc.items())
                    print(f"{label} {col} value_counts (top {max_rows}): {vc_txt}")

    def _log_step4_lines(block_label, merged_df):
        if merged_df is None or getattr(merged_df, "empty", True):
            return
        sub = merged_df
        if "Blokk" in merged_df.columns:
            sub = merged_df[merged_df["Blokk"] == block_label]
            if sub.empty:
                return
        n_val = None
        if "N" in sub.columns:
            n_series = pd.to_numeric(sub["N"], errors="coerce").dropna()
            if not n_series.empty:
                n_val = int(n_series.iloc[0])
        if n_val is None:
            n_val = len(sub)
        for _, row in sub.iloc[:4].iterrows():
            prof = row.get("Profil")
            mean = pd.to_numeric(pd.Series([row.get("Mean")]), errors="coerce").iloc[0]
            std = pd.to_numeric(pd.Series([row.get("Std")]), errors="coerce").iloc[0]
            print(
                f"[STEG 4] {block_label} mean_{prof}={float(mean):.6f} "
                f"std_{prof}={float(std):.6f} N={n_val}"
            )

    def _log_dirichlet_fit_rows(dir_res, dir_fits, cap=200):
        if dir_fits is None or getattr(dir_fits, "empty", True):
            return
        block_col = "Blokk" if "Blokk" in dir_fits.columns else ("block" if "block" in dir_fits.columns else None)
        ark_col = "Ark" if "Ark" in dir_fits.columns else ("sheet_name" if "sheet_name" in dir_fits.columns else ("source" if "source" in dir_fits.columns else None))
        if block_col is None or ark_col is None:
            return

        def _fmt_num(val):
            try:
                v = float(val)
            except Exception:
                return None
            if not np.isfinite(v):
                return None
            return f"{v:.6f}"

        keys = dir_fits[[block_col, ark_col]].drop_duplicates().itertuples(index=False, name=None)
        count = 0
        for block, ark in keys:
            if count >= cap:
                break
            fit_row = dir_fits[(dir_fits[block_col] == block) & (dir_fits[ark_col] == ark)]
            if fit_row.empty:
                continue
            fit_row = fit_row.iloc[0]
            alpha_vals = []
            if dir_res is not None and not getattr(dir_res, "empty", True):
                block_col_res = "Blokk" if "Blokk" in dir_res.columns else ("block" if "block" in dir_res.columns else None)
                ark_col_res = "Ark" if "Ark" in dir_res.columns else ("sheet_name" if "sheet_name" in dir_res.columns else ("source" if "source" in dir_res.columns else None))
                if block_col_res and ark_col_res:
                    sub_res = dir_res[(dir_res[block_col_res] == block) & (dir_res[ark_col_res] == ark)]
                    if not sub_res.empty and "alpha" in sub_res.columns:
                        for v in pd.to_numeric(sub_res["alpha"], errors="coerce").tolist():
                            alpha_vals.append(v)
            alpha_txt = "[" + ", ".join(f"{float(v):.6f}" if np.isfinite(v) else "nan" for v in alpha_vals) + "]"
            parts = [f"[DIR_FITROW] block={block}", f"source={ark}"]
            for key, col in [
                ("n_obs", "N_used"),
                ("S", "S"),
                ("LL_full", "LL_full"),
                ("LR", "LR"),
                ("p", "p"),
                ("AICc", "AICc_full"),
                ("R2_LR", "R2_LR"),
            ]:
                if col in dir_fits.columns:
                    val = _fmt_num(fit_row.get(col))
                    if val is not None:
                        parts.append(f"{key}={val}")
            parts.append(f"alpha={alpha_txt}")
            print(" ".join(parts))
            count += 1

    def _log_df(label, df, preview_rows=5, index=False):
        _log_df_meta(label, df)

    def _redact_list(vals):
        return "<REDACTED: REDACT_LOGS=1>" if REDACT_LOGS else vals

    
    
    # --- Konsollhjelpere (kompakte, trygge) ---
    def _print_table(name, df, n=8, rnd=4):
        _log_df_meta(name, df)

    SIGROW_CAP_DEFAULT = 200
    
    
    def _log_sig_summary(df, label, step_tag, cap=SIGROW_CAP_DEFAULT):
        p_col = None
        if "p" in df.columns:
            p_col = "p"
        elif "p_value" in df.columns:
            p_col = "p_value"
        if p_col is None:
            print(f"[SIG] step={step_tag} label={label} p_col=missing")
            return
    
        work = df.copy()
        work["_p"] = pd.to_numeric(work[p_col], errors="coerce")
        work = work.dropna(subset=["_p"])
    
        term_col = None
        if "term" in work.columns:
            term_col = "term"
        elif "clr_term" in work.columns:
            term_col = "clr_term"
    
        if term_col is not None:
            term_s = work[term_col].astype(str).str.strip().str.lower()
            nonconst = ~term_s.isin(["const", "intercept", "(intercept)"])
        else:
            nonconst = pd.Series(True, index=work.index)
    
        work_nc = work.loc[nonconst]
        sig_nc = work_nc[work_nc["_p"] < 0.05]
    
        print(
            f"[SIG] step={step_tag} label={label} "
            f"sig_count_nonconst={len(sig_nc)} total_nonconst={len(work_nc)}"
        )
    
        if "variant" in work_nc.columns and "response" in work_nc.columns:
            grp = (
                sig_nc.groupby(["variant", "response"], dropna=False)
                .size()
            .sort_values(ascending=False)
            .iloc[:15]
            )
            print(f"[SIG] step={step_tag} label={label} sig_by_variant_response")
            for (variant, response), count in grp.items():
                print(
                    f"  variant={variant} response={response} sig_count={count}"
                )
    
        top_cols = [c for c in ["variant", "response", term_col, "coef", "_p", "robust_cov_used"] if c]
        top_cols = [c for c in top_cols if c in work_nc.columns]
        if top_cols:
            print(f"[SIG] step={step_tag} label={label} top10_lowest_p")
            for _, row in work_nc.nsmallest(10, "_p")[top_cols].iterrows():
                kv = " ".join(f"{c}={row[c]}" for c in top_cols)
                print(f"  {kv}")
    
        sig_rows = sig_nc.copy()
        if sig_rows.empty:
            return
    
        sort_keys = ["_p"]
        for extra_key in ["variant", "response", term_col]:
            if extra_key and extra_key in sig_rows.columns:
                sort_keys.append(extra_key)
        sig_rows = sig_rows.sort_values(sort_keys)
    
        extra_id_cols = []
        for col in ["x_var", "y_var", "lhs", "rhs", "var1", "var2", "x", "y"]:
            if col in sig_rows.columns and col not in extra_id_cols:
                extra_id_cols.append(col)
    
        for i, (_, row) in enumerate(sig_rows.iterrows()):
            if i >= cap:
                print(
                    f"[SIG] step={step_tag} label={label} "
                    f"sigrow_truncated=true cap={cap} total_sig={len(sig_rows)}"
                )
                break
            fields = {
                "variant": row.get("variant"),
                "response": row.get("response"),
                "term": row.get(term_col) if term_col else None,
                "coef": row.get("coef"),
                "p": row.get("_p"),
                "robust": row.get("robust_cov_used"),
            }
            for col in extra_id_cols:
                fields[col] = row.get(col)
            parts = []
            for key, val in fields.items():
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                parts.append(f"{key}={val}")
            print(
                f"[SIGROW] step={step_tag} label={label} " + " ".join(parts)
            )

    def _fmt_num_log(val, digits=6):
        try:
            v = float(val)
        except Exception:
            return "NA"
        if not np.isfinite(v):
            return "NA"
        if digits == 6:
            return f"{v:.6g}"
        return f"{v:.{digits}f}"

    def _split_model_label(label: str):
        if label is None:
            return "NA", "NA"
        lab = str(label).strip()
        if " | " in lab:
            left, right = lab.split(" | ", 1)
            return left.strip(), right.strip() or "NA"
        if "(" in lab and ")" in lab:
            left, right = lab.split("(", 1)
            variant = right.split(")", 1)[0].strip()
            return left.strip(), variant or "NA"
        return lab, "NA"

    def _infer_blocks_from_family(model_family: str):
        mf = (model_family or "").strip()
        mf_low = mf.lower()
        if mf_low.startswith("strategy"):
            return "Strategi", "OCAI"
        if mf_low.startswith("ocai"):
            return "OCAI", "Strategi"
        return "NA", "NA"

    def _make_model_id(step_tag, model_family, variant, response):
        return f"{step_tag}|{model_family}|{variant}|{response}"

    def _fit_stats_dict(res):
        def _safe(getter, default=float("nan")):
            try:
                v = getter()
            except Exception:
                return default
            return v

        return {
            "nobs": _safe(lambda: getattr(res, "nobs")),
            "k": _safe(lambda: len(getattr(res, "params", []))),
            "df_model": _safe(lambda: getattr(res, "df_model")),
            "df_resid": _safe(lambda: getattr(res, "df_resid")),
            # F/pF kan kaste ValueError under robuste kovarianser (HC3/HC1) -> må være safe
            "F": _safe(lambda: getattr(res, "fvalue")),
            "pF": _safe(lambda: getattr(res, "f_pvalue")),
            "R2": _safe(lambda: getattr(res, "rsquared")),
            "R2_adj": _safe(lambda: getattr(res, "rsquared_adj")),
            "AIC": _safe(lambda: getattr(res, "aic")),
            "BIC": _safe(lambda: getattr(res, "bic")),
        }


    def _log_modeldef(step_tag, model_id, model_family, variant, response, y_block, x_block, robust_type, stats, x_terms):
        x_terms_txt = ",".join(map(str, x_terms)) if x_terms is not None else ""
        print(
            f"[MODELDEF] step={step_tag} model_id={model_id} family={model_family} variant={variant} "
            f"response={response} y_block={y_block} x_block={x_block} robust={robust_type} "
            f"nobs={_fmt_num_log(stats.get('nobs'))} k={_fmt_num_log(stats.get('k'))} "
            f"df_model={_fmt_num_log(stats.get('df_model'))} df_resid={_fmt_num_log(stats.get('df_resid'))} "
            f"F={_fmt_num_log(stats.get('F'))} pF={_fmt_num_log(stats.get('pF'))} "
            f"R2={_fmt_num_log(stats.get('R2'))} R2_adj={_fmt_num_log(stats.get('R2_adj'))} "
            f"AIC={_fmt_num_log(stats.get('AIC'))} BIC={_fmt_num_log(stats.get('BIC'))} "
            f"x_terms={x_terms_txt}"
        )

    def _log_fitrow(step_tag, model_id, model_family, variant, response, robust_type, stats):
        print(
            f"[FITROW] step={step_tag} model_id={model_id} family={model_family} variant={variant} "
            f"response={response} robust={robust_type} nobs={_fmt_num_log(stats.get('nobs'))} "
            f"k={_fmt_num_log(stats.get('k'))} df_model={_fmt_num_log(stats.get('df_model'))} "
            f"df_resid={_fmt_num_log(stats.get('df_resid'))} F={_fmt_num_log(stats.get('F'))} "
            f"pF={_fmt_num_log(stats.get('pF'))} R2={_fmt_num_log(stats.get('R2'))} "
            f"R2_adj={_fmt_num_log(stats.get('R2_adj'))} AIC={_fmt_num_log(stats.get('AIC'))} "
            f"BIC={_fmt_num_log(stats.get('BIC'))}"
        )

    def _log_coefrows(step_tag, model_id, model_family, variant, response, terms, res, robust_type, stats):
        params = np.asarray(getattr(res, "params", []), dtype=float)
        bse = np.asarray(getattr(res, "bse", []), dtype=float)
        tvals = np.asarray(getattr(res, "tvalues", []), dtype=float)
        pvals = np.asarray(getattr(res, "pvalues", []), dtype=float)
        for j, term in enumerate(terms):
            coef = params[j] if j < len(params) else np.nan
            se = bse[j] if j < len(bse) else np.nan
            tval = tvals[j] if j < len(tvals) else np.nan
            pval = pvals[j] if j < len(pvals) else np.nan
            print(
                f"[COEFROW] step={step_tag} model_id={model_id} family={model_family} variant={variant} "
                f"response={response} robust={robust_type} nobs={_fmt_num_log(stats.get('nobs'))} "
                f"k={_fmt_num_log(stats.get('k'))} df_model={_fmt_num_log(stats.get('df_model'))} "
                f"df_resid={_fmt_num_log(stats.get('df_resid'))} F={_fmt_num_log(stats.get('F'))} "
                f"pF={_fmt_num_log(stats.get('pF'))} R2={_fmt_num_log(stats.get('R2'))} "
                f"R2_adj={_fmt_num_log(stats.get('R2_adj'))} AIC={_fmt_num_log(stats.get('AIC'))} "
                f"BIC={_fmt_num_log(stats.get('BIC'))} term={term} coef={_fmt_num_log(coef)} "
                f"se={_fmt_num_log(se)} t={_fmt_num_log(tval)} p={_fmt_num_log(pval)}"
            )

    def _log_p_allrows(df, step_tag, label, cap=1000):
        if df is None or getattr(df, "empty", True):
            print(f"[PALL] step={step_tag} label={label} empty=1")
            return
        if "ID" in df.columns:
            print(f"[PALL] step={step_tag} label={label} SKIP: has_ID_col=1")
            return
        p_col = None
        for cand in ["p", "p_value", "pval", "pval_adj"]:
            if cand in df.columns:
                p_col = cand
                break
        if p_col is None:
            print(f"[PALL] step={step_tag} label={label} p_col=missing")
            return

        work = df.copy()
        work["_p"] = pd.to_numeric(work[p_col], errors="coerce")
        sort_keys = ["_p"]
        for extra in ["dataset", "Blokk", "block", "variant", "response", "test", "group", "term", "Profil", "x_var", "y_var"]:
            if extra in work.columns:
                sort_keys.append(extra)
        work = work.sort_values(sort_keys, na_position="last", kind="mergesort")

        p_series = work["_p"].dropna()
        def _fmt_num(val):
            try:
                v = float(val)
            except Exception:
                return "nan"
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6g}"

        p_min = p_series.min() if not p_series.empty else np.nan
        p_med = p_series.median() if not p_series.empty else np.nan
        p_max = p_series.max() if not p_series.empty else np.nan
        print(
            f"[PALL] step={step_tag} label={label} "
            f"n_rows={len(work)} p_min={_fmt_num(p_min)} p_med={_fmt_num(p_med)} p_max={_fmt_num(p_max)}"
        )

        key_cols_priority = [
            "dataset", "group_var", "Blokk", "block", "variant", "response", "test", "group", "Profil",
            "term", "x_var", "y_var", "rho", "stat", "df1", "df2", "U", "F", "chi2", "q",
            "N", "N_used", "conclusion",
        ]
        key_cols = [c for c in key_cols_priority if c in work.columns]

        for i, (_, row) in enumerate(work.iterrows()):
            if i >= cap:
                print(
                    f"[PALL] step={step_tag} label={label} "
                    f"pallrow_truncated=true cap={cap} total_rows={len(work)}"
                )
                break
            def _fmt_val(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return None
                if isinstance(val, (float, int, np.floating, np.integer)):
                    return _fmt_num(val)
                return str(val)

            fields = {}
            for col in key_cols:
                v = _fmt_val(row.get(col))
                if v is not None:
                    fields[col] = v
            key = "|".join(f"{k}={v}" for k, v in fields.items()) if fields else "row"
            parts = [f"key={key}"]
            for k, v in fields.items():
                parts.append(f"{k}={v}")
            parts.append(f"p={_fmt_num(row.get('_p'))}")
            print(f"[PALLROW] step={step_tag} label={label} " + " ".join(parts))

    def _log_icc_summary(df, step_tag="STEG5", label="ICC_SUMMARY", cap=1000):
        if df is None or getattr(df, "empty", True):
            print(f"[ICC] step={step_tag} label={label} empty=1")
            return
        if "ID" in df.columns:
            print(f"[ICC] step={step_tag} label={label} SKIP: has_ID_col=1")
            return

        icc_col = None
        for c in df.columns:
            if "icc" in str(c).lower():
                icc_col = c
                break
        if icc_col is None and "ICC1" in df.columns:
            icc_col = "ICC1"
        if icc_col is None:
            print(f"[ICC] step={step_tag} label={label} icc_col=missing")
            return

        work = df.copy()
        work["_icc"] = pd.to_numeric(work[icc_col], errors="coerce")

        def _fmt_num(val):
            try:
                v = float(val)
            except Exception:
                return "nan"
            if not np.isfinite(v):
                return "nan"
            return f"{v:.6g}"

        icc_series = work["_icc"].dropna()
        icc_min = icc_series.min() if not icc_series.empty else np.nan
        icc_med = icc_series.median() if not icc_series.empty else np.nan
        icc_max = icc_series.max() if not icc_series.empty else np.nan
        print(
            f"[ICC] step={step_tag} label={label} n_rows={len(work)} "
            f"icc_min={_fmt_num(icc_min)} icc_med={_fmt_num(icc_med)} icc_max={_fmt_num(icc_max)}"
        )

        blk_col = "Blokk" if "Blokk" in work.columns else ("block" if "block" in work.columns else None)
        prof_col = "Profil" if "Profil" in work.columns else ("profile" if "profile" in work.columns else None)
        dept_col = "Departement" if "Departement" in work.columns else ("Department" if "Department" in work.columns else None)

        if blk_col and prof_col:
            grp = work.groupby([blk_col, prof_col], dropna=False)["_icc"]
            summary = grp.agg(["mean", "median", "min", "max", "count"]).reset_index()
            summary = summary.sort_values([blk_col, prof_col], kind="mergesort")
            for _, row in summary.iterrows():
                print(
                    f"[ICC] step={step_tag} label={label} "
                    f"block={row.get(blk_col)} profile={row.get(prof_col)} "
                    f"mean={_fmt_num(row.get('mean'))} med={_fmt_num(row.get('median'))} "
                    f"min={_fmt_num(row.get('min'))} max={_fmt_num(row.get('max'))} n={int(row.get('count'))}"
                )

        sort_keys = [c for c in [blk_col, prof_col, dept_col] if c]
        if sort_keys:
            work = work.sort_values(sort_keys, kind="mergesort")
        for i, (_, row) in enumerate(work.iterrows()):
            if i >= cap:
                print(
                    f"[ICC] step={step_tag} label={label} "
                    f"iccrow_truncated=true cap={cap} total_rows={len(work)}"
                )
                break
            parts = []
            if blk_col:
                parts.append(f"block={row.get(blk_col)}")
            if prof_col:
                parts.append(f"profile={row.get(prof_col)}")
            if dept_col:
                parts.append(f"dept={row.get(dept_col)}")
            parts.append(f"icc={_fmt_num(row.get('_icc'))}")
            print(f"[ICCROW] step={step_tag} label={label} " + " ".join(parts))
    # --- Robust kovarians hjelper (HC3 -> HC1 fallback) ---
    _robust_fallback_counter = {"count": 0}

    def _get_robust_cov_with_fallback(m, prefer="HC3", fallback="HC1"):
        # k kan skilles fea len(X_cols) hvis tilpasset modell droppet/la til params (f.eks., kollineæritet/konst behandling)
        try:
            k = int(np.asarray(getattr(m, "params", [])).shape[0])
        except Exception:
            k = 0
        if k == 0:
            try:
                k = int(np.asarray(m.model.exog).shape[1])
            except Exception:
                k = 0

        def _try_cov(cov_type):
            reasons = []
            try:
                m_rob = m.get_robustcov_results(cov_type=cov_type)
            except Exception:
                return None, None, None, None, None, [f"{cov_type}_get_robustcov_failed"]
            try:
                cov = np.asarray(m_rob.cov_params(), dtype=float)
            except Exception:
                cov = None
            try:
                se = np.asarray(m_rob.bse, dtype=float)
            except Exception:
                se = None

            if cov is None:
                reasons.append(f"{cov_type}_cov_none")
            if se is None:
                reasons.append(f"{cov_type}_se_none")

            if k == 0 and cov is not None and cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
                k_eff = int(cov.shape[0])
            else:
                k_eff = k

            if cov is not None and (cov.ndim != 2 or (k_eff and cov.shape != (k_eff, k_eff))):
                reasons.append(f"{cov_type}_cov_bad_shape")
            if se is not None and (k_eff and se.shape != (k_eff,)):
                reasons.append(f"{cov_type}_se_bad_shape")
            if cov is not None and not np.isfinite(cov).all():
                reasons.append(f"{cov_type}_cov_nonfinite")
            if cov is not None and cov.ndim == 2:
                diag = np.diag(cov)
                if not np.isfinite(diag).all():
                    reasons.append(f"{cov_type}_cov_diag_nonfinite")
                if np.any(diag < 0):
                    reasons.append(f"{cov_type}_cov_diag_negative")
            if se is not None:
                if not np.isfinite(se).all():
                    reasons.append(f"{cov_type}_se_nonfinite")
                if np.all(np.isinf(se)):
                    reasons.append(f"{cov_type}_se_all_inf")
                if np.any(se == 0):
                    reasons.append(f"{cov_type}_se_zero")

            if reasons:
                return None, None, None, None, None, reasons

            try:
                tvals = np.asarray(m_rob.tvalues, dtype=float)
            except Exception:
                tvals = None
            try:
                pvals = np.asarray(m_rob.pvalues, dtype=float)
            except Exception:
                pvals = None
            if tvals is None or pvals is None:
                return None, None, None, None, None, [f"{cov_type}_t_p_missing"]

            return m_rob, cov, se, tvals, pvals, []

        m_rob, cov, se, tvals, pvals, reasons = _try_cov(prefer)
        if not reasons:
            return m_rob, cov, se, tvals, pvals, prefer, [], True

        m_rob_fb, cov_fb, se_fb, tvals_fb, pvals_fb, reasons_fb = _try_cov(fallback)
        if not reasons_fb:
            _robust_fallback_counter["count"] += 1
            return m_rob_fb, cov_fb, se_fb, tvals_fb, pvals_fb, fallback, reasons, True

        reasons.extend(reasons_fb)
        return None, None, None, None, None, "NONE", reasons, False
    
    
    
    # --- Konfig / stioppsett for Codex-miljøet ---
    
    # Basemappe = der denne fila ligger, dvs. .../Masteroppgave/code_codex
    BASE_DIR = Path(__file__).resolve().parent
    
    # Kjøremodus: DEV (syntetisk, default) eller REAL (krever eksplisitt DATA_DIR/DATA_FILE)
    RUN_MODE = str(os.getenv("RUN_MODE", "DEV")).strip().upper()
    if RUN_MODE not in {"DEV", "REAL"}:
        raise RuntimeError(f"Ugyldig RUN_MODE='{RUN_MODE}'. Bruk DEV eller REAL.")
    
    # SIKT-sikkerhet: i code_codex skal vi KUN bruke syntetiske data
    if "code_codex" in BASE_DIR.parts and RUN_MODE != "DEV":
        raise RuntimeError(
            "Forsøk på å bruke REAL DATA i code_codex. "
            "Dette er ikke tillatt. Bruk en separat realrun-mappe."
        )

    def _find_repo_root(start: Path) -> Optional[Path]:
        """Gå oppover fra start og finn mappe som inneholder .git."""
        cur = start.resolve()
        for _ in range(50):
            if (cur / ".git").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return None
    
    # Kildekatalog for data (syntetisk i DEV, eksplisitt i REAL)
    if RUN_MODE == "DEV":
        SYN_DATA_DIR = BASE_DIR / "synthetic_data"
        SYN_DATA_DIR.mkdir(exist_ok=True)  # trygghet hvis mappen mangler
        data_dir = SYN_DATA_DIR
        # Valg av hvilket syntetisk datasett vi vil jobbe mot:
        #   - 'eksempeldatasett.xlsx' (N=250)
        #   - 'eksempeldatasett_synthetic_mle.xlsx' (N=1000)
        DATA_FILE = "eksempeldatasett_synthetic_mle.xlsx"  # bytt til 250-varianten ved behov
    else:
        data_dir_env = os.getenv("DATA_DIR", "").strip()
        data_file_env = os.getenv("DATA_FILE", "").strip()
        if not data_dir_env or not data_file_env:
            raise RuntimeError(
                "RUN_MODE=REAL krever DATA_DIR og DATA_FILE som miljøvariabler."
            )
        data_dir = Path(data_dir_env).expanduser().resolve()
        repo_root = _find_repo_root(BASE_DIR)
        if repo_root is not None:
            try:
                data_dir.relative_to(repo_root)
                raise RuntimeError(
                    "RUN_MODE=REAL: DATA_DIR ligger inne i git-repoet "
                    f"({repo_root}). Dette er ikke tillatt. "
                    "Bruk en mappe utenfor repoet."
                )
            except ValueError:
                pass
        if "code_codex" in data_dir.parts:
            raise RuntimeError(
                "DATA_DIR peker inn i code_codex. "
                "Bruk en separat realrun-mappe utenfor code_codex."
            )
        DATA_FILE = data_file_env

    # ------------------------------------------
    # Tee stdout/stderr to a run log (optional)
    # ------------------------------------------
    if str(os.getenv("LOG_TEE", "1")).strip() != "0":
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_file_stem = Path(DATA_FILE).stem if DATA_FILE else ""
            if data_file_stem:
                log_name = f"{data_file_stem}_runlog_{RUN_MODE}_{ts}.log"
            else:
                log_name = f"runlog_{RUN_MODE}_{ts}.log"
            log_path = Path(data_dir) / log_name

            class _Tee:
                def __init__(self, stream, fh):
                    self._stream = stream
                    self._fh = fh
                def write(self, s):
                    self._stream.write(s)
                    try:
                        self._fh.write(s)
                    except Exception:
                        pass
                    return len(s)
                def flush(self):
                    try:
                        self._stream.flush()
                    except Exception:
                        pass
                    try:
                        self._fh.flush()
                    except Exception:
                        pass
                def isatty(self):
                    return getattr(self._stream, "isatty", lambda: False)()

            _log_fh = open(log_path, "a", encoding="utf-8", errors="replace")
            _orig_stdout = sys.stdout
            _orig_stderr = sys.stderr
            sys.stdout = _Tee(sys.stdout, _log_fh)
            sys.stderr = _Tee(sys.stderr, _log_fh)

            def _restore_streams():
                try:
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    except Exception:
                        pass
                    sys.stdout = _orig_stdout
                    sys.stderr = _orig_stderr
                finally:
                    try:
                        _log_fh.close()
                    except Exception:
                        pass

            atexit.register(_restore_streams)
            print(f"[LOG] capturing console output to {log_path}")
            register_output(step="INIT", label="RUNLOG", path=str(log_path), kind="log", note="LOG_TEE")
        except Exception:
            # Fail open: keep normal printing if log setup fails
            pass
    
    xlsx_path = data_dir / DATA_FILE
    base_path = xlsx_path
    pbg_path = base_path.with_name(base_path.stem + "_p+bg.xlsx")
    data_file_stem = Path(DATA_FILE).stem if DATA_FILE else "run"
    def _write_manifest():
        try:
            write_output_manifest(data_dir, data_file_stem, RUN_MODE)
        except Exception:
            pass
    atexit.register(_write_manifest)
    print(f"[MODE] RUN_MODE={RUN_MODE} | data_dir={data_dir}")
    print(f"[INPUT] Hoveddata: {xlsx_path}")
    print(f"[INPUT] p+bg-fil (skrive-/lesesti): {pbg_path}")
    
    
    ocai_cols = ["Klan", "Adhockrati", "Marked", "Hierarki"]
    strat_cols = ["Opportunist", "Entreprenør", "Spekulant", "Konservativ"]
    
    targets = [
        ("OCAI - dominerende", ocai_cols),
        ("OCAI - strategiske", ocai_cols),
        ("OCAI - suksess", ocai_cols),
        ("Strategi - dominerende", strat_cols),
        ("Strategi - strategiske", strat_cols),
        ("Strategi - suksess", strat_cols),
    ]
    
    def find_sheet_name(prefix: str, available: list[str]) -> str | None:
        """Finn første ark der navnet starter med prefix (case-insensitive)."""
        for s in available:
            if s.lower().startswith(prefix.lower()):
                return s
        return None
    
    
    # Kontrollerer datakvalitet: beregner reliabilitet på 'Kontroll'-arket (A–E) og kjører Little's MCAR per ark med numeriske kolonner.
    # Bruker xlsx_path som input og skriver bare konsollrapporter (ingen filer) om reliabilitet og mangler.
    # ============================
    # STEG X: Reliabilitet ('Kontroll') + Little's MCAR (per ark)
    # Forutsetter: xlsx_path er definert over.
    # ============================
    
    def _cronbach_alpha(dfz: pd.DataFrame) -> float:
        """Cronbach's alpha på z-skalerte items; krever minst to kolonner og returnerer NaN om ikke beregnbar."""
        # Standardize (z) so variances are 1
        Z = (dfz - dfz.mean()) / dfz.std(ddof=1)
        k = Z.shape[1]
        if k < 2:
            return np.nan
        item_vars = Z.var(ddof=1)
        total_var = Z.sum(axis=1).var(ddof=1)
        if total_var == 0:
            return np.nan
        return (k / (k - 1.0)) * (1.0 - item_vars.sum() / total_var)
    
    def _omega_total_pca(dfz: pd.DataFrame) -> float:
        """McDonald's omega (total) basert på enfaktor-PCA av korrelasjonsmatrisen."""
        # McDonald's omega (total) via 1-faktor PCA på korrelasjonsmatrise
        Z = (dfz - dfz.mean()) / dfz.std(ddof=1)
        R = np.corrcoef(Z.values, rowvar=False)
        # beskytt mot numeriske småavvik
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        # egenverdier/-vektorer
        vals, vecs = np.linalg.eigh(R)
        # største egenverdi og tilhørende vektor
        idx = np.argmax(vals)
        lam1 = float(vals[idx])
        v1 = vecs[:, idx]
        # lastinger (enfaktor) for standardiserte items
        loadings = np.sqrt(max(lam1, 0.0)) * v1
        # unike varianskomponenter (psi) på korrelasjonsnivå
        psi = 1.0 - loadings**2
        num = (np.sum(loadings))**2
        den = num + np.sum(psi)
        return float(num / den) if den > 0 else np.nan
    
    def _split_half_even_odd(dfz: pd.DataFrame):
        """Deler items i partall/oddetall, korrelerer totalskårer og beregner Spearman–Brown-korrigert reliabilitet."""
        # del items på indeks: partall vs oddetall
        cols = list(dfz.columns)
        even_cols = [c for i,c in enumerate(cols) if i % 2 == 0]
        odd_cols  = [c for i,c in enumerate(cols) if i % 2 == 1]
        if len(even_cols) == 0 or len(odd_cols) == 0:
            return np.nan, np.nan
        Z = (dfz - dfz.mean()) / dfz.std(ddof=1)
        x = Z[even_cols].sum(axis=1)
        y = Z[odd_cols].sum(axis=1)
        if x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
            return np.nan, np.nan
        r = float(np.corrcoef(x, y)[0,1])
        sb = (2*r)/(1+r) if (1+r) != 0 else np.nan
        return r, sb
    
    def _little_mcar_test_approx(df_num: pd.DataFrame):
        """
        En praktisk Little’s MCAR-implementasjon (tilnærmet):
        - Deler inn i mønstre etter hvilke kolonner som er observerte.
        - Bruker moments (mean/cov) fra listwise complete som referanse.
        - χ² ≈ sum_p n_p * (μ_p - μ)_O' Σ_OO^{-1} (μ_p - μ)_O  over mønstre p.
        - df ≈ sum_p |O_p| - q, q=len(alle kolonner).
        NB: Dette er en vanlig, pragmatisk implementasjon; for helt eksakt EM-versjon, bruk egen pakke.
        """
        q = df_num.shape[1]
        if q < 2:
            return None
    
        # Fullt komplette rader for referanse (mean/cov)
        cc = df_num.dropna()
        if cc.shape[0] < 5:
            return None
    
        mu = cc.mean().values
        Sigma = cc.cov().values
    
        # Basis-sjekk: singulær?
        try:
            np.linalg.cholesky(Sigma + 1e-8*np.eye(q))
        except np.linalg.LinAlgError:
            return {"skipped": "Singulær kovarians fra komplette rader."}
    
        # Mønstre (mask av observert=True/False)
        mask = ~df_num.isna()
        patterns, inv, counts = np.unique(mask.values, axis=0, return_inverse=True, return_counts=True)
    
        chi2 = 0.0
        df_sum = 0
        used_patterns = 0
    
        for p_idx, pat in enumerate(patterns):
            idx_rows = np.where(inv == p_idx)[0]
            n_p = int(counts[p_idx])
            O = np.where(pat)[0]
            if n_p == 0 or len(O) == 0 or len(O) == q:
                # Hvis alle observert (komplett) → bidrar ikke til test (μ_p ≈ μ),
                # eller hvis 0 observert, ubrukelig.
                continue
    
            sub_cols = df_num.columns[O]
            sub = df_num.loc[idx_rows, sub_cols].dropna(axis=0, how="any")
            if sub.shape[0] == 0:
                continue
    
            mu_p = sub.mean().values
            mu_O = mu[O]
            Sigma_OO = Sigma[np.ix_(O, O)]
    
            # invertibelt?
            try:
                inv_S = np.linalg.inv(Sigma_OO)
            except np.linalg.LinAlgError:
                # hopp over mønstre med singulær sub-kovarians
                continue
    
            d = (mu_p - mu_O)
            stat = float(n_p * d.T.dot(inv_S).dot(d))
            chi2 += stat
            df_sum += len(O)
            used_patterns += 1
    
        # frihetsgrader: sum(|O_p|) - q (Little, 1988)
        df = df_sum - q
        if used_patterns == 0 or df <= 0:
            return {"skipped": "Ingen gyldige mønstre (for få mangler/bare komplette)."}
        pval = 1.0 - _chi2.cdf(chi2, df)
        return {"chi2": chi2, "df": int(df), "p": float(pval), "patterns_used": int(used_patterns)}
    
    # ---------- (1) Reliabilitet på 'Kontroll' ----------
    print("\n[STEG X | Reliabilitet – 'Kontroll']")
    print(f"Fil: {xlsx_path.name}")
    
    try:
        kontroll = pd.read_excel(xlsx_path, sheet_name="Kontroll")
    except Exception as e:
        print(f"Kunne ikke lese arket 'Kontroll': {e}")
        kontroll = None
    
    if kontroll is not None:
        # Anta at Likert-variablene heter A–E; tilpass om nødvendig
        likert_cols = ["A","B","C","D","E"]
        missing_cols = [c for c in likert_cols if c not in kontroll.columns]
        if missing_cols:
            print(f"Fant ikke kolonner i 'Kontroll': {missing_cols}")
        else:
            sub = kontroll[likert_cols].apply(pd.to_numeric, errors="coerce")
            # listwise for reliabilitetsmål
            sub_listwise = sub.dropna()
            N_used = sub_listwise.shape[0]
            print(f"Ark: Kontroll  |  Variabler: {likert_cols}  |  N brukt (listwise): {N_used}")
            if N_used >= 5 and sub_listwise.shape[1] >= 2:
                alpha = _cronbach_alpha(sub_listwise)
                omega = _omega_total_pca(sub_listwise)
                r_half, sb_half = _split_half_even_odd(sub_listwise)
                print(f"Cronbach's alpha: {alpha:.4f}")
                print(f"McDonald's omega (total, PCA-1faktor): {omega:.4f}")
                if np.isnan(r_half):
                    print("Split-half: ikke beregnbar (for få items eller nullvariasjon).")
                else:
                    print(f"Split-half r (even–odd): {r_half:.4f}  → Spearman–Brown: {sb_half:.4f}")
            else:
                print("For lite data til reliabilitet (trenger ≥5 rader og ≥2 items etter listwise).")
    
    # ---------- (2) Little's MCAR per ark (kun der det gir mening) ----------
    print("\n[STEG X | Little's MCAR – per ark med relevante numeriske variabler]")
    
    try:
        xls = pd.ExcelFile(xlsx_path)
        sheet_names = xls.sheet_names
    except Exception as e:
        print(f"Kunne ikke åpne arbeidsboken: {e}")
        sheet_names = []
    
    for sh in sheet_names:
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sh)
        except Exception as e:
            print(f"- {sh}: kunne ikke lese ({e})")
            continue
    
        # plukk numeriske kolonner
        num_df = df.select_dtypes(include=[np.number]).copy()
        # caster strings som ser ut som tall
        for c in df.columns.difference(num_df.columns):
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().any() and coerced.isna().any():
                num_df[c] = coerced
    
        # behold bare kolonner med minst litt variasjon
        if num_df.shape[1] < 2:
            print(f"- {sh}: hopper over (færre enn 2 numeriske kolonner).")
            continue
    
        na_cols = [c for c in num_df.columns if num_df[c].isna().any()]
        if len(na_cols) == 0:
            print(f"- {sh}: hopper over (0 kolonner med mangler).")
            continue
    
        # dropp helt konstante kolonner
        keep = [c for c in num_df.columns if num_df[c].dropna().std(ddof=1) > 0]
        num_df = num_df[keep]
        if num_df.shape[1] < 2:
            print(f"- {sh}: hopper over (konstante kolonner etter rensing).")
            continue
    
        res = _little_mcar_test_approx(num_df)
        if res is None:
            print(f"- {sh}: hopper over (for få komplette rader til referanse).")
        elif "skipped" in res:
            print(f"- {sh}: hopper over ({res['skipped']})")
        else:
            print(f"- {sh}: Little’s MCAR χ²={res['chi2']:.2f}, df={res['df']}, p={res['p']:.4f}  "
                  f"(mønstre brukt: {res['patterns_used']})")
    
    
    # Sjekker at hver profilrad i målearjene summerer til 100 og rapporterer avvik.
    # Input er xlsx-arkene definert i targets, output er en konsolltabell med OK/feilstatuser.
    # =======================
    # STEG 1: SUM-KONTROLL
    # =======================
    # - Leser alle relevante ark (targets)
    # - Sjekker at profiler (4 kolonner) finnes (tillater case-variant)
    # - Summer per rad skal være 100
    # - Skriver konsoll-oversikt
    
    xls = pd.ExcelFile(xlsx_path)
    summary = []
    
    for prefix, cols in targets:
        sh = find_sheet_name(prefix, xls.sheet_names)
        if sh is None:
            summary.append({"sheet_match": prefix, "sheet_found": None, "status": "ARK MANGLER"})
            continue
    
        df = pd.read_excel(xlsx_path, sheet_name=sh)
    
        # eksakt match eller case-insensitive fallback
        if set(cols).issubset(df.columns):
            use_cols = cols
        else:
            use_cols = []
            lc = {c.lower(): c for c in df.columns}
            for c in cols:
                if c.lower() in lc:
                    use_cols.append(lc[c.lower()])
            if len(use_cols) != 4:
                summary.append({
                    "sheet_match": prefix,
                    "sheet_found": sh,
                    "status": "KOLONNER MANGLER",
                    "have": list(df.columns)
                })
                continue
    
        # summer og sjekk
        sums = df[use_cols].sum(axis=1, numeric_only=True)
        bad = (sums != 100)
    
        summary.append({
            "sheet_match": prefix,
            "sheet_found": sh,
            "status": "OK" if bad.sum() == 0 else "RADER!=100",
            "rows": int(len(df)),
            "bad_rows": int(bad.sum()),
            "min_sum": float(sums.min()) if len(sums) else None,
            "max_sum": float(sums.max()) if len(sums) else None
        })
    
    summary_df = pd.DataFrame(summary)
    print("\n--- SUM-KONTROLL ---")
    _log_df_meta("SUM-KONTROLL", summary_df)

    aitchison_validate = str(os.getenv("AITCHISON_VALIDATE", "")).strip().lower() in ("1", "true", "yes", "on")
    aitchison_validate_announced = False

    def _aggregate_profiles_aitchison_per_id(out: pd.DataFrame, cols: list[str],
                                             block: str, stage_label: str,
                                             expected_rows_per_id: int = 3) -> pd.DataFrame:
        """
        Felles Aitchison-aggregator med konsistenskontroller.
        - Krever expected_rows_per_id rader per ID før aggregering.
        - Bruker CLR-mean (log-geometri) med eps og closure.
        - Verifiserer én rad per ID og sum≈1 etterpå.
        - Skriver form/kolonner til konsoll for gjennomsiktighet.
        """
        if out is None or out.empty:
            return pd.DataFrame()

        df = out.copy()
        df["ID"] = df["ID"].astype(str)

        counts = df.groupby("ID").size()
        bad_counts = counts[counts != expected_rows_per_id]
        if not bad_counts.empty:
            raise RuntimeError(
                f"[{stage_label}] {block}: forventet {expected_rows_per_id} rader per ID før Aitchison-agg. "
                f"Avvik for {len(bad_counts)} ID-er."
            )

        print(f"[{stage_label}] {block}: råprofiler før Aitchison-agg "
              f"shape={df.shape}, kolonner={['ID'] + cols}")
        _log_df_meta("  meta:", df)

        eps = 1e-12
        X = df[cols].to_numpy(float)
        X = np.clip(X, eps, None)
        X = X / X.sum(axis=1, keepdims=True)

        logX = np.log(X)
        clr = logX - logX.mean(axis=1, keepdims=True)

        clr_df = pd.DataFrame(clr, columns=cols)
        clr_df.insert(0, "ID", df["ID"].to_numpy())
        clr_mean = clr_df.groupby("ID")[cols].mean()

        P_ait = np.exp(clr_mean.to_numpy(float))
        P_ait = P_ait / P_ait.sum(axis=1, keepdims=True)

        agg = clr_mean.reset_index()
        agg[cols] = P_ait
        agg["ID"] = agg["ID"].astype(str)

        if agg["ID"].duplicated().any():
            dup_ids = agg["ID"][agg["ID"].duplicated()].unique().tolist()
            raise RuntimeError(
                f"[{stage_label}] {block}: dupliserte ID-er etter aggregering "
                f"(n={len(dup_ids)})"
            )

        agg_sums = agg[cols].sum(axis=1)
        bad_sum = agg_sums[(agg_sums - 1.0).abs() > 1e-6]
        if not bad_sum.empty:
            raise RuntimeError(
                f"[{stage_label}] {block}: komposisjoner summerer ikke til 1 etter closure for {len(bad_sum)} ID-er "
                f"(antall={len(bad_sum)})."
            )
        if (agg[cols] <= 0).any().any():
            raise RuntimeError(f"[{stage_label}] {block}: fant ikke-positive deler etter aggregering.")

        print(f"[{stage_label}] {block}: etter Aitchison-agg shape={agg.shape}, "
              f"sum≈1 i [min={agg_sums.min():.6f}, max={agg_sums.max():.6f}]")
        _log_df_meta("  meta:", agg)

        if aitchison_validate:
            nonlocal aitchison_validate_announced
            if not aitchison_validate_announced:
                print("[STEG2] AITCHISON_VALIDATE=ON (env var)")
                aitchison_validate_announced = True
            # Validering: Aitchison-agg (CLR-mean) vs. geometrisk mean per komponent + closure
            logX_df = pd.DataFrame(logX, columns=cols)
            logX_df.insert(0, "ID", df["ID"].to_numpy())
            log_mean = logX_df.groupby("ID")[cols].mean()
            P_geo = np.exp(log_mean.to_numpy(float))
            P_geo = P_geo / P_geo.sum(axis=1, keepdims=True)
            diff = np.abs(P_ait - P_geo)
            per_id_max = diff.max(axis=1)
            overall_max = float(diff.max()) if diff.size else np.nan
            if per_id_max.size:
                print(
                    f"[{stage_label}] {block}: Aitchison-validate n_ids={len(per_id_max)}, "
                    f"overall_max={overall_max:.3e}, "
                    f"per_id_max[min={float(np.min(per_id_max)):.3e}, "
                    f"median={float(np.median(per_id_max)):.3e}, "
                    f"max={float(np.max(per_id_max)):.3e}]"
                )
            if np.isfinite(overall_max) and overall_max > 1e-12:
                print(f"[{stage_label}] {block}: ADVARSEL Aitchison-validate overall_max={overall_max:.3e} "
                      f"(kan skyldes eps/closure-rekkefølge eller ikke-ekvivalent implementasjon).")

        return agg
    
    # Bygger p-tabeller (0..1) fra profilark og slår inn bakgrunnsopplysninger via ID fra et felles 'Bakgrunn'-ark.
    # Tar inn xlsx_path/targets og produserer p_tables og with_bg_tables i minnet som grunnlag for videre analyser.
    # =======================
    # STEG 2 (NY): P-MATRISER + BAKGRUNN fra eget ark ("Bakgrunn")
    # =======================
    # - Leser ett felles bakgrunnsark ("Bakgrunn" / "Background" prefiks, case-insensitivt)
    # - Slår bakgrunn inn i hver måltabell via ID (left merge)
    # - Skalerer profiler til p (0..1)
    # - Lager:
    #     p_tables[arknavn] = DataFrame(profiler i 0..1)
    #     with_bg_tables[arknavn] = {"p": <bg + p>, "orig": <bg + originale 0..100 profiler>}
    
    def _load_background_df(xlsx_path: Path) -> pd.DataFrame:
        """Leser felles bakgrunnsark (Bakgrunn/Background), normaliserer ID/kategorier og returnerer ev. tom DF hvis mangler."""
        xls = pd.ExcelFile(xlsx_path)
        # tillat både "Bakgrunn" og "Background"
        sh_bg = (find_sheet_name("Bakgrunn", xls.sheet_names)
                 or find_sheet_name("Background", xls.sheet_names))
        if sh_bg is None:
            print("[STEG 2] Fant ikke 'Bakgrunn'-ark. Fortsetter uten sammenslåing (ingen bg).")
            return pd.DataFrame()
    
        bg = pd.read_excel(xlsx_path, sheet_name=sh_bg)
    
        # Normaliser navn + typer
        cols_keep = ["ID", "Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]
        bg = bg[[c for c in cols_keep if c in bg.columns]].copy()
    
        if "ID" not in bg.columns:
            print(f"[STEG 2] Bakgrunnsark '{sh_bg}' mangler kolonnen 'ID' – kan ikke merge. Hopper over bg.")
            return pd.DataFrame()
      
        # Rydd typer
        bg["ID"] = bg["ID"].astype(str)
        for c in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]:
            if c in bg.columns:
                # behold streng for kategoriske; tall der det gir mening
                if c in ["Alder", "Ansiennitet"]:
                    # prøv numerisk, ellers streng
                    cand = pd.to_numeric(bg[c], errors="coerce")
                    if cand.notna().any():
                        bg[c] = cand
                    else:
                        bg[c] = bg[c].astype("string")
                else:
                    bg[c] = bg[c].astype("string")
        return bg
    
    # --- bygg p-tabeller + merge bakgrunn ---
    p_tables: dict[str, pd.DataFrame] = {}
    with_bg_tables: dict[str, dict[str, pd.DataFrame]] = {}
    p_tables_id: dict[str, pd.DataFrame] = {}
    with_bg_tables_id: dict[str, dict[str, pd.DataFrame]] = {}
    _block_raw: dict[str, list[pd.DataFrame]] = {"OCAI": [], "Strategi": []}
    _block_sheets: dict[str, list[str]] = {"OCAI": [], "Strategi": []}
    _block_agg: dict[str, pd.DataFrame] = {}
    
    xls = pd.ExcelFile(xlsx_path)
    BG = _load_background_df(xlsx_path)  # kan være tom
    
    for prefix, cols in targets:
        sh = find_sheet_name(prefix, xls.sheet_names)
        if sh is None:
            print(f"[HOPPER OVER] Finner ikke ark som starter med: {prefix}")
            continue
    
        df = pd.read_excel(xlsx_path, sheet_name=sh)
    
        # finn de 4 profilkolonnene (eksakt el. case-insensitivt)
        if set(cols).issubset(df.columns):
            use_cols = cols
        else:
            lc = {c.lower(): c for c in df.columns}
            use_cols = [lc[c.lower()] for c in cols if c.lower() in lc]
    
        if len(use_cols) != 4:
            print(f"[HOPPER OVER] Kolonner mangler i {sh}. Har: {list(df.columns)}")
            continue
    
        # p-matrise 0..1
        P = df[use_cols].astype(float) / 100.0
        p_tables[sh] = P.copy()

        # ID-baserte p-profiler (krav: én rad per ID per ark)
        block = "OCAI" if prefix.lower().startswith("ocai") else "Strategi"
        if "ID" in df.columns:
            P_id = df[["ID"] + use_cols].copy()
            P_id["ID"] = P_id["ID"].astype(str)
            for c in use_cols:
                P_id[c] = pd.to_numeric(P_id[c], errors="coerce") / 100.0
            if P_id["ID"].isna().any():
                missing_id_count = int(P_id["ID"].isna().sum())
                raise RuntimeError(f"[STEG 2] {block}/{sh}: manglende ID-er: {missing_id_count}.")
            dup_ids = P_id["ID"][P_id["ID"].duplicated()].unique().tolist()
            if dup_ids:
                raise RuntimeError(f"[STEG 2] {block}/{sh}: dupliserte ID-er i enkeltark: {len(dup_ids)}.")
            if P_id[use_cols].isna().any().any():
                bad_count = int(P_id[use_cols].isna().any(axis=1).sum())
                raise RuntimeError(f"[STEG 2] {block}/{sh}: NaN i profiler før blokk-agg: {bad_count} rader.")
            sums = P_id[use_cols].sum(axis=1)
            bad_sum = (sums - 1.0).abs() > 1e-6
            if bad_sum.any():
                bad_count = int(bad_sum.sum())
                raise RuntimeError(f"[STEG 2] {block}/{sh}: radsum != 1 før blokk-agg: {bad_count} rader.")
            print(f"[STEG 2] {block}/{sh}: path={xlsx_path}, shape={P_id.shape}, cols={list(P_id.columns)}")
            print(f"[STEG 2] {block}/{sh}: rows={len(P_id)}, unique_IDs={P_id['ID'].nunique()}, dup_IDs={len(dup_ids)}")
            _log_df_meta("  meta:", P_id)
            _block_raw[block].append(P_id)
            _block_sheets[block].append(sh)
        else:
            print(f"[STEG 2] Advarsel: '{sh}' mangler 'ID' – kan ikke lage ID-aggregert p-tabell.")
    
        # bygg original (0..100) + p med bakgrunn fra eget ark (via ID)
        if "ID" in df.columns and not BG.empty:
            # normaliser ID til streng på begge før merge
            df_ID = df[["ID"]].copy()
            df_ID["ID"] = df_ID["ID"].astype(str)
    
            # slå sammen én gang, left på ID (bevar rekkefølge)
            merged_bg = df_ID.merge(BG, on="ID", how="left")
    
            # informer hvis mange mangler
            miss = merged_bg["ID"][merged_bg.drop(columns=["ID"]).isna().all(axis=1)].shape[0]
            if miss > 0:
                print(f"[STEG 2] Advarsel: {miss} rad(er) i '{sh}' uten match i 'Bakgrunn'.")
    
            # bygg p+bg og orig+bg
            p_bg = pd.concat([merged_bg.reset_index(drop=True), P.reset_index(drop=True)], axis=1)
            orig_bg = pd.concat([merged_bg.reset_index(drop=True),
                                 df[use_cols].reset_index(drop=True)], axis=1)
            # ID-nivå håndteres etter blokk-agg (ikke her)
        else:
            # fallback: ingen ID eller ingen BG -> fortsatt uten bakgrunn (som før)
            if "ID" not in df.columns:
                print(f"[STEG 2] Advarsel: '{sh}' mangler 'ID' – kan ikke matche mot 'Bakgrunn'.")
            elif BG.empty:
                print(f"[STEG 2] Advarsel: 'Bakgrunn' ikke tilgjengelig – '{sh}' lagres uten bg.")
            p_bg = pd.concat([df.get(["ID"], pd.Series(index=df.index, dtype=object)),
                              P.reset_index(drop=True)], axis=1)
            orig_bg = pd.concat([df.get(["ID"], pd.Series(index=df.index, dtype=object)),
                                 df[use_cols].reset_index(drop=True)], axis=1)
            # ID-nivå håndteres etter blokk-agg (ikke her)
    
        with_bg_tables[sh] = {"p": p_bg, "orig": orig_bg}
    
    # Blokk-agg per ID (3 ark per blokk)
    for block in ["OCAI", "Strategi"]:
        if not _block_raw[block]:
            continue
        block_stack = pd.concat(_block_raw[block], axis=0, ignore_index=True)
        counts = block_stack.groupby("ID").size()
        print(f"[STEG 2] {block}: blokk-stacket shape={block_stack.shape}, cols={list(block_stack.columns)}")
        _log_counts_summary(f"[STEG 2] {block}: per-ID counts (summary)", counts)
        if not (counts == 3).all():
            bad_counts = counts[counts != 3]
            raise RuntimeError(
                f"[STEG 2] {block}: forventet 3 rader per ID etter blokk-stacking. "
                f"Avvik for {len(bad_counts)} ID-er."
            )
        cols = ocai_cols if block == "OCAI" else strat_cols
        block_agg = _aggregate_profiles_aitchison_per_id(
            block_stack[["ID"] + cols], cols, block, stage_label=f"STEG2/{block}_BLOCK"
        )
        _block_agg[block] = block_agg.copy()
        print(f"[STEG 2] {block}: blokk-agg shape={block_agg.shape}, cols={list(block_agg.columns)}")
        _log_df_meta("  meta:", block_agg)

        for sh in _block_sheets[block]:
            p_tables_id[sh] = block_agg.copy()

        if not BG.empty:
            block_agg = block_agg.copy()
            block_agg["ID"] = block_agg["ID"].astype(str)
            bg_id = BG.copy()
            bg_id["ID"] = bg_id["ID"].astype(str)
            merged_bg_id = block_agg[["ID"]].merge(bg_id, on="ID", how="left")
            if merged_bg_id["ID"].duplicated().any():
                dup_ids = merged_bg_id["ID"][merged_bg_id["ID"].duplicated()].unique().tolist()
                raise RuntimeError(f"[STEG 2] {block}: dupliserte ID-er etter BG-merge: {len(dup_ids)}.")
            if len(merged_bg_id) != len(block_agg):
                raise RuntimeError(
                    f"[STEG 2] {block}: uventet radtap etter BG-merge "
                    f"(før={len(block_agg)}, etter={len(merged_bg_id)})."
                )
            miss_id = merged_bg_id["ID"][merged_bg_id.drop(columns=["ID"]).isna().all(axis=1)].shape[0]
            if miss_id > 0:
                print(f"[STEG 2] Advarsel: {miss_id} ID-er i blokk '{block}' uten match i 'Bakgrunn' (ID-agg).")
            p_bg_id_block = pd.concat([merged_bg_id.reset_index(drop=True),
                                       block_agg[cols].reset_index(drop=True)], axis=1)
            for sh in _block_sheets[block]:
                with_bg_tables_id[sh] = {"p": p_bg_id_block.copy(), "orig": block_agg.copy()}

    print(f"[STEG 2] Bygget {len(p_tables)} p-tabeller og {len(with_bg_tables)} p+bakgrunn-tabeller (fra eget 'Bakgrunn'-ark).")
    print(f"[STEG 2] Bygget {len(p_tables_id)} ID-aggregert p-tabeller og {len(with_bg_tables_id)} ID-aggregert p+bakgrunn-tabeller.")
    
    # Lagrer alle p+bakgrunn-tabeller fra forrige steg til én Excel-fil (ett ark per skjema) med CSV-fallback.
    # Input er with_bg_tables/pbg_path, output er en lagret _p+bg.xlsx-fil eller CSV-er.
    # =======================
    # STEG 3: LAGRING p+bg (med fallback)
    # =======================
    # - Skriver hvert ark i with_bg_tables som et eget Excel-ark "<originalt arknavn> (p)"
    # - Faller tilbake til openpyxl hvis xlsxwriter feiler
    # - Hvis begge feiler, lagres én CSV per ark
    
    out_p = pbg_path  # <eksempeldatasett>_p+bg.xlsx
    out_p_id = pbg_path.with_name(pbg_path.stem + "_ID.xlsx")
    pbg_path_id = out_p_id
    
    try:
        with pd.ExcelWriter(out_p, engine="xlsxwriter") as writer:
            for sh, d in with_bg_tables.items():
                base = f"{sh} (p)"
                sheet_name = base[:31]  # Excel-begrensning
                export_excel(d["p"], writer=writer, sheet_name=sheet_name, label=sheet_name)
        print(f"[STEG 3] Lagret p+bakgrunn til: {out_p}")
        register_output(step="STEG 3", label="p+bg", path=out_p, kind="xlsx")
        for sh, d in with_bg_tables.items():
            print(f"[STEG 3] p+bg (stacked) {sh}: path={out_p}, shape={d['p'].shape}, cols={list(d['p'].columns)}")
            _log_df_meta("  meta:", d["p"])
            _log_small_cats("  cats:", d["p"])
    except Exception as e:
        print(f"[STEG 3] xlsxwriter feilet ({e!r}). Prøver openpyxl ...")
        try:
            with pd.ExcelWriter(out_p, engine="openpyxl") as writer:
                for sh, d in with_bg_tables.items():
                    base = f"{sh} (p)"
                    sheet_name = base[:31]
                    export_excel(d["p"], writer=writer, sheet_name=sheet_name, label=sheet_name)
            print(f"[STEG 3] Lagret p+bakgrunn til: {out_p} (openpyxl)")
            register_output(step="STEG 3", label="p+bg", path=out_p, kind="xlsx", note="openpyxl")
            for sh, d in with_bg_tables.items():
                print(f"[STEG 3] p+bg (stacked) {sh}: path={out_p}, shape={d['p'].shape}, cols={list(d['p'].columns)}")
                _log_df_meta("  meta:", d["p"])
                _log_small_cats("  cats:", d["p"])
        except Exception as e2:
            print(f"[STEG 3] Excel-skriving feilet ({e2!r}). Lagrer CSV i stedet.")
            for sh, d in with_bg_tables.items():
                csv_safe = sh.replace(" ", "_").replace("/", "_")
                csv_path = xlsx_path.with_name(f"{xlsx_path.stem}_p+bg__{csv_safe}.csv")
                d["p"].to_csv(csv_path, index=False)
                print(f"[STEG 3] Lagret CSV: {csv_path}")
                register_output(step="STEG 3", label=f"p+bg_{sh}", path=csv_path, kind="csv", df=d["p"], note="xlsx_fail")
                print(f"[STEG 3] p+bg (stacked) {sh}: path={csv_path}, shape={d['p'].shape}, cols={list(d['p'].columns)}")
                _log_df_meta("  meta:", d["p"])
                _log_small_cats("  cats:", d["p"])

    # Skriv ID-aggregert p+bg
    if with_bg_tables_id:
        try:
            with pd.ExcelWriter(out_p_id, engine="xlsxwriter") as writer:
                for sh, d in with_bg_tables_id.items():
                    base = f"{sh} (p_ID)"
                    sheet_name = base[:31]
                    export_excel(d["p"], writer=writer, sheet_name=sheet_name, label=sheet_name)
            print(f"[STEG 3] Lagret ID-aggregert p+bakgrunn til: {out_p_id}")
            register_output(step="STEG 3", label="p+bg_ID", path=out_p_id, kind="xlsx")
            for sh, d in with_bg_tables_id.items():
                print(f"[STEG 3] p+bg (ID) {sh}: path={out_p_id}, shape={d['p'].shape}, cols={list(d['p'].columns)}")
                _log_df_meta("  meta:", d["p"])
                _log_small_cats("  cats:", d["p"])
        except Exception as e:
            print(f"[STEG 3] xlsxwriter feilet for ID-agg ({e!r}). Prøver openpyxl ...")
            try:
                with pd.ExcelWriter(out_p_id, engine="openpyxl") as writer:
                    for sh, d in with_bg_tables_id.items():
                        base = f"{sh} (p_ID)"
                        sheet_name = base[:31]
                        export_excel(d["p"], writer=writer, sheet_name=sheet_name, label=sheet_name)
                print(f"[STEG 3] Lagret ID-aggregert p+bakgrunn til: {out_p_id} (openpyxl)")
                register_output(step="STEG 3", label="p+bg_ID", path=out_p_id, kind="xlsx", note="openpyxl")
                for sh, d in with_bg_tables_id.items():
                    print(f"[STEG 3] p+bg (ID) {sh}: path={out_p_id}, shape={d['p'].shape}, cols={list(d['p'].columns)}")
                    _log_df_meta("  meta:", d["p"])
                    _log_small_cats("  cats:", d["p"])
            except Exception as e2:
                print(f"[STEG 3] Excel-skriving feilet for ID-agg ({e2!r}). Lagrer CSV i stedet.")
                for sh, d in with_bg_tables_id.items():
                    csv_safe = sh.replace(" ", "_").replace("/", "_")
                    csv_path = xlsx_path.with_name(f"{xlsx_path.stem}_p+bg_ID__{csv_safe}.csv")
                    d["p"].to_csv(csv_path, index=False)
                    print(f"[STEG 3] Lagret CSV (ID-agg): {csv_path}")
                    register_output(step="STEG 3", label=f"p+bg_ID_{sh}", path=csv_path, kind="csv", df=d["p"], note="xlsx_fail")
                    print(f"[STEG 3] p+bg (ID) {sh}: path={csv_path}, shape={d['p'].shape}, cols={list(d['p'].columns)}")
                    _log_df_meta("  meta:", d["p"])
                    _log_small_cats("  cats:", d["p"])
    
    # Samler alle OCAI- og Strategi-profiler på tvers av ark, beregner Mean/Std og skriver en oppsummeringsfil.
    # Input er p_tables fra steg 2, output er <dataset>_summary_overall.xlsx + konsollvisning.
    # =======================
    # STEG 4: SAMLET OCAI/STRATEGI (MERGET) – Mean og Std
    # =======================
    # - Slår sammen alle p-tabeller per blokk (OCAI / Strategi)
    # - Beregner Mean og Std (ddof=1) for hver profil
    # - Skriver til <eksempeldatasett>_summary_overall.xlsx
    # - Skriver kort konsoll-rapport
    
    ocai_frames: list[pd.DataFrame] = []
    strat_frames: list[pd.DataFrame] = []
    
    if "OCAI" in _block_agg:
        ocai_frames.append(_block_agg["OCAI"][ocai_cols].copy())
    if "Strategi" in _block_agg:
        strat_frames.append(_block_agg["Strategi"][strat_cols].copy())
    
    ocai_merged = pd.DataFrame()
    strat_merged = pd.DataFrame()
    
    if ocai_frames:
        ocai_cat = pd.concat(ocai_frames, axis=0, ignore_index=True)
        ocai_mean = ocai_cat.mean().rename("Mean")
        ocai_std  = ocai_cat.std(ddof=1).rename("Std")
        ocai_merged = (
            pd.concat([ocai_mean, ocai_std], axis=1)
            .reset_index()
            .rename(columns={"index": "Profil"})
        )
        ocai_merged.insert(0, "Blokk", "OCAI")
        ocai_merged.insert(1, "N", int(len(ocai_cat)))
    
    if strat_frames:
        strat_cat = pd.concat(strat_frames, axis=0, ignore_index=True)
        strat_mean = strat_cat.mean().rename("Mean")
        strat_std  = strat_cat.std(ddof=1).rename("Std")
        strat_merged = (
            pd.concat([strat_mean, strat_std], axis=1)
            .reset_index()
            .rename(columns={"index": "Profil"})
        )
        strat_merged.insert(0, "Blokk", "Strategi")
        strat_merged.insert(1, "N", int(len(strat_cat)))
    
    out_summary = xlsx_path.with_name(xlsx_path.stem + "_summary_overall_ID.xlsx")
    with pd.ExcelWriter(out_summary, engine="xlsxwriter") as writer:
        if not ocai_merged.empty:
            export_excel(ocai_merged, writer=writer, sheet_name="OCAI_merged_ID", label="OCAI_merged_ID")
        if not strat_merged.empty:
            export_excel(strat_merged, writer=writer, sheet_name="Strategi_merged_ID", label="Strategi_merged_ID")

    print(f"[STEG 4] Skrev samlet sammendrag til: {out_summary}")
    register_output(step="STEG 4", label="summary_overall_ID", path=out_summary, kind="xlsx")
    if not ocai_merged.empty:
        print(f"[STEG 4] OCAI: path={out_summary}, shape={ocai_merged.shape}, cols={list(ocai_merged.columns)}")
        _log_df_meta("[STEG 4] OCAI (meta)", ocai_merged)
        _log_step4_lines("OCAI", ocai_merged)
    if not strat_merged.empty:
        print(f"[STEG 4] Strategi: path={out_summary}, shape={strat_merged.shape}, cols={list(strat_merged.columns)}")
        _log_df_meta("[STEG 4] Strategi (meta)", strat_merged)
        _log_step4_lines("Strategi", strat_merged)
    if not ocai_merged.empty:
        _log_df_meta("[OCAI samlet] Mean og Std (meta)", ocai_merged)
    if not strat_merged.empty:
        _log_df_meta("[Strategi samlet] Mean og Std (meta)", strat_merged)
    
    # Leser p+bg-filen og estimerer ICC(1) per profil gruppert på 'Departement' for OCAI og Strategi.
    # Input er pbg_path og profilkolonner; output er en ICC-tabell på Excel/CSV og kort konsollrapport.
    # =======================
    # STEG 5: ICC(1) PER DEPARTEMENT
    # =======================
    # - Leser p+bg-filen fra STEG 3 (ark som ender med " (p)")
    # - Regner ICC(1) per profil innenfor hver blokk (OCAI/Strategi) gruppert på 'Departement'
    # - Skriver tabell til <eksempeldatasett>_icc.xlsx (fallback til CSV)
    
    def _pick_icc_sheet(xls_local: pd.ExcelFile, block: str):
        block_key = block.lower()
        def _is_block_sheet(sh: str, require_suffix: bool = True) -> bool:
            sh_l = sh.lower()
            if require_suffix and not sh_l.endswith(" (p_id)"):
                return False
            if block_key == "ocai":
                return sh_l.startswith("ocai")
            return sh_l.startswith("strategi") or sh_l.startswith("strategy") or sh_l.startswith("strat")

        with_suffix = [n for n in xls_local.sheet_names if _is_block_sheet(n, require_suffix=True)]
        candidates = with_suffix or [n for n in xls_local.sheet_names if _is_block_sheet(n, require_suffix=False)]
        if not candidates:
            return None, [], "ingen kandidatark for blokk"
        prefer = [n for n in candidates if "departement" in n.lower()]
        chosen = prefer[0] if prefer else candidates[0]
        return chosen, candidates, ""

    def _collect_block_df_with_dept(pbg_path: Path, block: str,
                                    ocai_cols: list[str], strat_cols: list[str],
                                    sheet_name: str | None = None) -> pd.DataFrame:
        """Henter p+bg-ark for gitt blokk, sikrer skala 0..1 og returnerer profiler med Departement."""
        xls_local = pd.ExcelFile(pbg_path)
        chosen = sheet_name
        if chosen is None:
            chosen, _, _ = _pick_icc_sheet(xls_local, block)
        if not chosen:
            return pd.DataFrame()
        df = pd.read_excel(pbg_path, sheet_name=chosen)

        # Valg av riktige profiler for blokken
        prof = [c for c in (ocai_cols if block == "OCAI" else strat_cols) if c in df.columns]
        if not prof or "Departement" not in df.columns:
            return pd.DataFrame()

        keep = ["Departement"] + prof
        out = df[keep].copy()
    
        # skaler til 0..1 hvis nødvendig
        profs = [c for c in out.columns if c != "Departement"]
        if profs:
            out[profs] = out[profs].apply(pd.to_numeric, errors="coerce")
            if out[profs].max().max() > 1.0:
                out[profs] = out[profs] / 100.0
    
        # streng-kategori for departement
        out["Departement"] = out["Departement"].astype("string")
        return out
    
    
    def _icc_oneway(y: pd.Series, groups: pd.Series):
        """
        Enveis ICC(1) (random effects, one-way):
          ICC1 = (MS_between - MS_within) / (MS_between + (mbar - 1)*MS_within)
        Returnerer (icc1, N, k, ms_between, ms_within)
        """
        y = pd.Series(y).astype(float)
        g = pd.Series(groups).astype("string")
        mask = y.notna() & g.notna()
        y = y[mask]; g = g[mask]
        if y.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan
    
        group_means = y.groupby(g).mean()
        group_sizes = y.groupby(g).size().astype(float)
        grand_mean = y.mean()
        k = float(len(group_means))
        N = float(len(y))
        if k < 2 or N <= k:
            return np.nan, N, k, np.nan, np.nan
    
        ss_between = float(((group_means - grand_mean) ** 2 * group_sizes).sum())
        ss_within = float((y.groupby(g).apply(lambda s: ((s - s.mean()) ** 2).sum())).sum())
    
        df_between = k - 1.0
        df_within = N - k
        ms_between = ss_between / df_between if df_between > 0 else np.nan
        ms_within = ss_within / df_within if df_within > 0 else np.nan
    
        mbar = float(group_sizes.mean())
        if np.isnan(ms_between) or np.isnan(ms_within):
            return np.nan, N, k, ms_between, ms_within
    
        icc1 = (ms_between - ms_within) / (ms_between + (mbar - 1.0) * ms_within)
        return icc1, N, k, ms_between, ms_within
    
    
    def _icc_table_for_block(pbg_path: Path, block: str, profiles: list[str], sheet_name: str | None = None) -> pd.DataFrame:
        """Bygger ICC-tabell for en blokk ved å beregne ICC(1) per profil fra p+bg-filen."""
        df_block = _collect_block_df_with_dept(pbg_path, block, ocai_cols, strat_cols, sheet_name=sheet_name)
        rows = []
        if df_block.empty:
            return pd.DataFrame(columns=[
                "Blokk","Profil","N","Grupper","MeanGrpSize","MS_between","MS_within","ICC1"
            ])
    
        for col in profiles:
            if col not in df_block.columns:
                continue
            icc1, N, k, msb, msw = _icc_oneway(df_block[col], df_block["Departement"])
            mbar = (N / k) if (k and not np.isnan(k) and k > 0) else np.nan
            rows.append({
                "Blokk": block,
                "Profil": col,
                "N": int(N) if not np.isnan(N) else np.nan,
                "Grupper": int(k) if not np.isnan(k) else np.nan,
                "MeanGrpSize": float(mbar) if not np.isnan(mbar) else np.nan,
                "MS_between": float(msb) if msb is not None and not np.isnan(msb) else np.nan,
                "MS_within": float(msw) if msw is not None and not np.isnan(msw) else np.nan,
                "ICC1": float(icc1) if icc1 is not None and not np.isnan(icc1) else np.nan,
            })
        return pd.DataFrame(rows)
    
    
    ICC_COLS = ["Blokk","Profil","N","Grupper","MeanGrpSize","MS_between","MS_within","ICC1"]
    ICC_NUM_COLS = ["N","Grupper","MeanGrpSize","MS_between","MS_within","ICC1"]

    xls_icc = pd.ExcelFile(pbg_path_id)
    ocai_sheet, ocai_candidates, ocai_reason = _pick_icc_sheet(xls_icc, "OCAI")
    strat_sheet, strat_candidates, strat_reason = _pick_icc_sheet(xls_icc, "Strategi")
    print(f"[STEG 5][ICC] OCAI sheet valgt: {ocai_sheet}")
    print(f"[STEG 5][ICC] OCAI kandidater: {ocai_candidates}")
    print(f"[STEG 5][ICC] Strategi sheet valgt: {strat_sheet}")
    print(f"[STEG 5][ICC] Strategi kandidater: {strat_candidates}")

    if not ocai_sheet:
        print(f"[STEG 5][ICC][ADVARSEL] Ingen OCAI-ark funnet. Kandidater={ocai_candidates} | {ocai_reason}")
    if not strat_sheet:
        print(f"[STEG 5][ICC][ADVARSEL] Ingen Strategi-ark funnet. Kandidater={strat_candidates} | {strat_reason}")

    df_icc_ocai = _collect_block_df_with_dept(pbg_path_id, "OCAI", ocai_cols, strat_cols, sheet_name=ocai_sheet)
    df_icc_strat = _collect_block_df_with_dept(pbg_path_id, "Strategi", strat_cols, strat_cols, sheet_name=strat_sheet)
    print(f"[STEG 5][ICC] OCAI df shape={df_icc_ocai.shape}, cols={list(df_icc_ocai.columns)}")
    print(f"[STEG 5][ICC] Strategi df shape={df_icc_strat.shape}, cols={list(df_icc_strat.columns)}")

    icc_ocai = _icc_table_for_block(pbg_path_id, "OCAI", ocai_cols, sheet_name=ocai_sheet)
    icc_strat = _icc_table_for_block(pbg_path_id, "Strategi", strat_cols, sheet_name=strat_sheet)

    print(f"[STEG 5][ICC] icc_ocai shape={icc_ocai.shape} cols={list(icc_ocai.columns)}")
    print(f"[STEG 5][ICC] icc_strat shape={icc_strat.shape} cols={list(icc_strat.columns)}")

    if icc_strat.empty:
        print("[STEG 5][ICC][ADVARSEL] ICC for Strategi er tom. Diagnostikk følger.")
        print(f"[STEG 5][ICC] Strategi df shape={df_icc_strat.shape}, cols={list(df_icc_strat.columns)}")
        print(f"[STEG 5][ICC] Strategi kandidater={strat_candidates} | valgt={strat_sheet} | {strat_reason}")
        if not df_icc_strat.empty:
            for c in strat_cols:
                if c in df_icc_strat.columns:
                    print(f"[STEG 5][ICC] Strategi non-NA {c}: {int(df_icc_strat[c].notna().sum())}")
            if "Departement" in df_icc_strat.columns:
                print(f"[STEG 5][ICC] Strategi Departement non-NA: {int(df_icc_strat['Departement'].notna().sum())}")

    def _ensure_icc_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reindex(columns=ICC_COLS)
        if not df.empty:
            df[ICC_NUM_COLS] = df[ICC_NUM_COLS].apply(pd.to_numeric, errors="coerce")
        return df

    icc_ocai = _ensure_icc_dtypes(icc_ocai)
    icc_strat = _ensure_icc_dtypes(icc_strat)

    _frames = [df for df in (icc_ocai, icc_strat) if not df.empty]
    if _frames:
        ICC_SUMMARY = pd.concat(_frames, axis=0, ignore_index=True)
    else:
        ICC_SUMMARY = pd.DataFrame(columns=ICC_COLS)
    
    print("\n--- ICC(1) per Departement ---")
    _log_df_meta("ICC(1) per Departement (meta)", ICC_SUMMARY)
    
    # Lagre til fil (Excel med fallback til CSV)
    out_icc = base_path.with_name(base_path.stem + "_icc_ID.xlsx")
    try:
        with pd.ExcelWriter(out_icc, engine="xlsxwriter") as writer:
            export_excel(ICC_SUMMARY, writer=writer, sheet_name="ICC_per_dept", label="ICC_per_dept")
        print(f"[STEG 5] Lagret ICC-tabell til: {out_icc}")
        register_output(step="STEG 5", label="ICC_per_dept", path=out_icc, kind="xlsx", df=ICC_SUMMARY)
        print(f"[STEG 5] path={out_icc}, shape={ICC_SUMMARY.shape}, cols={list(ICC_SUMMARY.columns)}")
        _log_df_meta("[STEG 5] ICC_SUMMARY (meta)", ICC_SUMMARY)
    except Exception:
        try:
            with pd.ExcelWriter(out_icc, engine="openpyxl") as writer:
                export_excel(ICC_SUMMARY, writer=writer, sheet_name="ICC_per_dept", label="ICC_per_dept")
            print(f"[STEG 5] Lagret ICC-tabell til: {out_icc} (openpyxl)")
            register_output(step="STEG 5", label="ICC_per_dept", path=out_icc, kind="xlsx", df=ICC_SUMMARY, note="openpyxl")
            print(f"[STEG 5] path={out_icc}, shape={ICC_SUMMARY.shape}, cols={list(ICC_SUMMARY.columns)}")
            _log_df_meta("[STEG 5] ICC_SUMMARY (meta)", ICC_SUMMARY)
        except Exception:
            out_csv = base_path.with_name(base_path.stem + "_icc_ID.csv")
            ICC_SUMMARY.to_csv(out_csv, index=False)
            print(f"[STEG 5] Excel-skriving feilet. Lagret som CSV: {out_csv}")
            register_output(step="STEG 5", label="ICC_per_dept", path=out_csv, kind="csv", df=ICC_SUMMARY, note="xlsx_fail")
            print(f"[STEG 5] path={out_csv}, shape={ICC_SUMMARY.shape}, cols={list(ICC_SUMMARY.columns)}")
            _log_df_meta("[STEG 5] ICC_SUMMARY (meta)", ICC_SUMMARY)

    _log_icc_summary(ICC_SUMMARY, "STEG5", "ICC_SUMMARY", cap=2000)
    
    # Slår sammen p+bg-ark og lager Mean/Std per profil innen hver bakgrunnsvariabel (departement, ansiennitet osv.).
    # Input er pbg_path/ocai_cols/strat_cols; output er en Excel med sammendrag per bakgrunn + konsollutdrag.
    # =======================
    # STEG 6: Sammendrag per bakgrunnsvariabel (Mean/Std) for OCAI og Strategi
    # =======================
    # - Leser p+bg-filen (fra STEG 3)
    # - Slår sammen alle relevante ark per blokk (bare faktiske bakgrunnskolonner beholdes)
    # - Lager tabeller (per bakgrunnsvariabel) med Mean/Std for hver profil
    # - Skriver til <eksempeldatasett>_by_category_summary.xlsx (fallback håndteres i except)
    
    def _load_merged_block_from_pbg(block: str, ocai_cols: list[str], strat_cols: list[str]) -> pd.DataFrame:
        """Laster og slår sammen alle (p)-ark for blokk, beholder faktiske bakgrunnskolonner og skalerer profiler ved behov."""
        xls = pd.ExcelFile(pbg_path)
        frames: list[pd.DataFrame] = []
        for sh in [n for n in xls.sheet_names if n.endswith(" (p)")]:
            df = pd.read_excel(pbg_path, sheet_name=sh)
    
            # Valg av profiler for blokk
            if block == "OCAI":
                profs = [c for c in ocai_cols if c in df.columns]
            else:
                profs = [c for c in strat_cols if c in df.columns]
            if not profs:
                continue
    
            # bakgrunn som faktisk finnes i arket
            bg_candidates = ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]
            bg_cols = [c for c in bg_candidates if c in df.columns]
            keep = bg_cols + profs
            frames.append(df[keep].copy())
    
        if not frames:
            return pd.DataFrame()
    
        out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
       
    
        # skaler profiler til 0..1 ved behov
        prof_cols = [c for c in out.columns if c in (ocai_cols + strat_cols)]
        if prof_cols and out[prof_cols].max().max() > 1.0:
            out[prof_cols] = out[prof_cols] / 100.0
    
        # sørg for str-kategori i bakgrunn
        for c in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]:
            if c in out.columns:
                out[c] = out[c].astype("string")
        return out
    
    
    def _by_category_tables(df_block: pd.DataFrame, profiles: list[str]) -> dict[str, pd.DataFrame]:
        """
        Returner dict{bg_var: DataFrame( Bakgrunn, Kategori, Profil, Mean, Std )}
        for alle bakgrunnsvariabler som finnes i df_block.
        """
        if df_block.empty:
            return {}
    
        tables: dict[str, pd.DataFrame] = {}
        for bg in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]:
            if bg not in df_block.columns:
                continue
            g = df_block.groupby(bg)[profiles]
            mean = g.mean()
            std = g.std(ddof=1)
    
            # lang-format med Mean og Std side om side
            mean_long = mean.stack().to_frame("Mean")
            std_long = std.stack().to_frame("Std")
            out = mean_long.join(std_long).reset_index()
            out = out.rename(columns={"level_1": "Profil", bg: "Kategori"})
            out.insert(0, "Bakgrunn", bg)
            tables[bg] = out
        return tables
    
    
    # 1) Last sammenslåtte blokker fra p+bg-filen (bruker kun faktisk bakgrunn i filen)
    def _load_merged_block_from_pbg_id(block: str, ocai_cols: list[str], strat_cols: list[str]) -> pd.DataFrame:
        """Laster og slår sammen alle (p_ID)-ark for blokk (ID-aggregert)."""
        xls = pd.ExcelFile(pbg_path_id)
        frames: list[pd.DataFrame] = []
        prefix = "ocai" if block == "OCAI" else "strategi"
        id_sheets = [n for n in xls.sheet_names if n.endswith(" (p_ID)") and n.lower().startswith(prefix)]
        if not id_sheets:
            # Fallback: håndter avkortede ark-navn som mister "(p_ID)"-suffixet.
            id_sheets = [n for n in xls.sheet_names if n.lower().startswith(prefix)]
        for sh in id_sheets:
            df = pd.read_excel(pbg_path_id, sheet_name=sh)
            # Valg av profiler for blokk
            if block == "OCAI":
                profs = [c for c in ocai_cols if c in df.columns]
            else:
                profs = [c for c in strat_cols if c in df.columns]
            if not profs:
                continue
            bg_candidates = ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]
            bg_cols = [c for c in bg_candidates if c in df.columns]
            keep = ["ID"] + bg_cols + profs
            frames.append(df[keep].copy())
            break
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=0, ignore_index=True, sort=False)
        prof_cols = [c for c in out.columns if c in (ocai_cols + strat_cols)]
        if prof_cols and out[prof_cols].max().max() > 1.0:
            out[prof_cols] = out[prof_cols] / 100.0
        for c in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]:
            if c in out.columns:
                out[c] = out[c].astype("string")
        if out["ID"].duplicated().any():
            dup_ids = out["ID"][out["ID"].duplicated()].unique().tolist()
            raise RuntimeError(f"[STEG 6] {block}: dupliserte ID-er i (p_ID): {_redact_list(dup_ids)}")
        return out

    _ocai_block = _load_merged_block_from_pbg_id("OCAI", ocai_cols, strat_cols)
    _strat_block = _load_merged_block_from_pbg_id("Strategi", ocai_cols, strat_cols)
    
    # 2) Lag tabeller per bakgrunnsvariabel
    ocai_tables = _by_category_tables(_ocai_block, [c for c in ocai_cols if c in _ocai_block.columns])
    strat_tables = _by_category_tables(_strat_block, [c for c in strat_cols if c in _strat_block.columns])

    # 2b) Likert (Kontroll) – rå ID-nivå per kategori
    def _load_kontroll_with_bg(xlsx_path: Path, bg_src: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        sh_ctrl = find_sheet_name("Kontroll", pd.ExcelFile(xlsx_path).sheet_names)
        if sh_ctrl is None:
            return pd.DataFrame(), [], []
        ctrl = pd.read_excel(xlsx_path, sheet_name=sh_ctrl)
        if "ID" not in ctrl.columns:
            return pd.DataFrame(), [], []
        ctrl = ctrl.copy()
        ctrl["ID"] = ctrl["ID"].astype(str)
        likert_cols = [c for c in ctrl.columns if c != "ID"]

        bg_cols = []
        if not bg_src.empty and "ID" in bg_src.columns:
            bg_src = bg_src.copy()
            bg_src["ID"] = bg_src["ID"].astype(str)
            bg_cols = [c for c in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"] if c in bg_src.columns]
            bg_keep = bg_src[["ID"] + bg_cols].drop_duplicates("ID")
            merged = bg_keep.merge(ctrl, on="ID", how="left")
        else:
            merged = ctrl
        return merged, likert_cols, bg_cols

    _bg_src = _ocai_block if not _ocai_block.empty else _strat_block
    _likert_df, _likert_cols, _likert_bg_cols = _load_kontroll_with_bg(xlsx_path, _bg_src)

    def _likert_summary_tables(df: pd.DataFrame, likert_cols: list[str], bg_cols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        if df.empty or not likert_cols:
            return pd.DataFrame(), {}
        rows = []
        for c in likert_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            rows.append({
                "Bakgrunn": "ALL",
                "Kategori": "ALL",
                "Profil": c,
                "Mean": float(s.mean()) if s.notna().any() else np.nan,
                "Std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
            })
        likert_all = pd.DataFrame(rows)

        tables: dict[str, pd.DataFrame] = {}
        for bg in bg_cols:
            if bg not in df.columns:
                continue
            g = df.groupby(bg, dropna=True)
            rows_bg = []
            for cat, sub in g:
                for c in likert_cols:
                    s = pd.to_numeric(sub[c], errors="coerce")
                    rows_bg.append({
                        "Bakgrunn": bg,
                        "Kategori": cat,
                        "Profil": c,
                        "Mean": float(s.mean()) if s.notna().any() else np.nan,
                        "Std": float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan,
                    })
            if rows_bg:
                tables[bg] = pd.DataFrame(rows_bg)
        return likert_all, tables

    likert_all, likert_tables = _likert_summary_tables(_likert_df, _likert_cols, _likert_bg_cols)
    
    # 3) Skriv til Excel – ett ark per bakgrunnsvariabel + samleark
    _bycat_path = base_path.with_name(base_path.stem + "_by_category_summary_ID.xlsx")

    try:
        ocai_sheet_list = (["OCAI_ALL_ID"] + [("OCAI_" + bg + "_ID")[:31] for bg in ocai_tables.keys()]) if ocai_tables else []
        strat_sheet_list = (["STRAT_ALL_ID"] + [("STRAT_" + bg + "_ID")[:31] for bg in strat_tables.keys()]) if strat_tables else []
        print(f"[STEG 6] Bycat_ID sheets (OCAI): {ocai_sheet_list}")
        print(f"[STEG 6] Bycat_ID sheets (STRAT): {strat_sheet_list}")
        with pd.ExcelWriter(_bycat_path, engine="xlsxwriter") as writer:
            # OCAI
            if ocai_tables:
                ocai_all = pd.concat(ocai_tables.values(), axis=0, ignore_index=True)
                export_excel(ocai_all, writer=writer, sheet_name="OCAI_ALL_ID", label="OCAI_ALL_ID")
                for bg, tbl in ocai_tables.items():
                    sheet = ("OCAI_" + bg + "_ID")[:31]
                    export_excel(tbl, writer=writer, sheet_name=sheet, label=sheet)

            # STRATEGI
            if strat_tables:
                strat_all = pd.concat(strat_tables.values(), axis=0, ignore_index=True)
                export_excel(strat_all, writer=writer, sheet_name="STRAT_ALL_ID", label="STRAT_ALL_ID")
                for bg, tbl in strat_tables.items():
                    sheet = ("STRAT_" + bg + "_ID")[:31]
                    export_excel(tbl, writer=writer, sheet_name=sheet, label=sheet)

            # LIKERT
            if not likert_all.empty:
                export_excel(likert_all, writer=writer, sheet_name="LIKERT_ALL_ID", label="LIKERT_ALL_ID")
                for bg, tbl in likert_tables.items():
                    sheet = ("LIKERT_" + bg + "_ID")[:31]
                    export_excel(tbl, writer=writer, sheet_name=sheet, label=sheet)

        print(f"[STEG 6] Lagret sammendrag per kategori til: {_bycat_path}")
        register_output(step="STEG 6", label="bycat_ID", path=_bycat_path, kind="xlsx")
        if ocai_tables:
            print(f"[STEG 6] OCAI_ALL: path={_bycat_path}, shape={ocai_all.shape}, cols={list(ocai_all.columns)}")
            _log_df_meta("[STEG 6] OCAI_ALL (meta)", ocai_all)
        if strat_tables:
            print(f"[STEG 6] STRAT_ALL: path={_bycat_path}, shape={strat_all.shape}, cols={list(strat_all.columns)}")
            _log_df_meta("[STEG 6] STRAT_ALL (meta)", strat_all)
    
    except Exception:
        # Fallback til openpyxl, deretter til CSV
        try:
            ocai_sheet_list = (["OCAI_ALL_ID"] + [("OCAI_" + bg + "_ID")[:31] for bg in ocai_tables.keys()]) if ocai_tables else []
            strat_sheet_list = (["STRAT_ALL_ID"] + [("STRAT_" + bg + "_ID")[:31] for bg in strat_tables.keys()]) if strat_tables else []
            print(f"[STEG 6] Bycat_ID sheets (OCAI): {ocai_sheet_list}")
            print(f"[STEG 6] Bycat_ID sheets (STRAT): {strat_sheet_list}")
            with pd.ExcelWriter(_bycat_path, engine="openpyxl") as writer:
                if ocai_tables:
                    ocai_all = pd.concat(ocai_tables.values(), axis=0, ignore_index=True)
                    export_excel(ocai_all, writer=writer, sheet_name="OCAI_ALL_ID", label="OCAI_ALL_ID")
                    for bg, tbl in ocai_tables.items():
                        sheet = ("OCAI_" + bg + "_ID")[:31]
                        export_excel(tbl, writer=writer, sheet_name=sheet, label=sheet)
                if strat_tables:
                    strat_all = pd.concat(strat_tables.values(), axis=0, ignore_index=True)
                    export_excel(strat_all, writer=writer, sheet_name="STRAT_ALL_ID", label="STRAT_ALL_ID")
                    for bg, tbl in strat_tables.items():
                        sheet = ("STRAT_" + bg + "_ID")[:31]
                        export_excel(tbl, writer=writer, sheet_name=sheet, label=sheet)
                if not likert_all.empty:
                    export_excel(likert_all, writer=writer, sheet_name="LIKERT_ALL_ID", label="LIKERT_ALL_ID")
                    for bg, tbl in likert_tables.items():
                        sheet = ("LIKERT_" + bg + "_ID")[:31]
                        export_excel(tbl, writer=writer, sheet_name=sheet, label=sheet)
            print(f"[STEG 6] Lagret sammendrag per kategori til: {_bycat_path} (openpyxl)")
            register_output(step="STEG 6", label="bycat_ID", path=_bycat_path, kind="xlsx", note="openpyxl")
            if ocai_tables:
                print(f"[STEG 6] OCAI_ALL: path={_bycat_path}, shape={ocai_all.shape}, cols={list(ocai_all.columns)}")
                _log_df_meta("[STEG 6] OCAI_ALL (meta)", ocai_all)
            if strat_tables:
                print(f"[STEG 6] STRAT_ALL: path={_bycat_path}, shape={strat_all.shape}, cols={list(strat_all.columns)}")
                _log_df_meta("[STEG 6] STRAT_ALL (meta)", strat_all)
        except Exception:
            print("[STEG 6] Excel-skriving feilet – lagrer CSV per tabell i samme mappe.")
            if ocai_tables:
                for bg, tbl in ocai_tables.items():
                    csv_path = _bycat_path.with_name(_bycat_path.stem + f"__OCAI_{bg}_ID.csv")
                    tbl.to_csv(csv_path, index=False)
                    print(f"[STEG 6] OCAI_{bg}: path={csv_path}, shape={tbl.shape}, cols={list(tbl.columns)}")
                    register_output(step="STEG 6", label=f"OCAI_{bg}_ID", path=csv_path, kind="csv", df=tbl, note="xlsx_fail")
                    _log_df_meta(f"[STEG 6] OCAI_{bg} (meta)", tbl)
            if strat_tables:
                for bg, tbl in strat_tables.items():
                    csv_path = _bycat_path.with_name(_bycat_path.stem + f"__STRAT_{bg}_ID.csv")
                    tbl.to_csv(csv_path, index=False)
                    print(f"[STEG 6] STRAT_{bg}: path={csv_path}, shape={tbl.shape}, cols={list(tbl.columns)}")
                    register_output(step="STEG 6", label=f"STRAT_{bg}_ID", path=csv_path, kind="csv", df=tbl, note="xlsx_fail")
                    _log_df_meta(f"[STEG 6] STRAT_{bg} (meta)", tbl)
            if not likert_all.empty:
                csv_path = _bycat_path.with_name(_bycat_path.stem + "__LIKERT_ALL_ID.csv")
                likert_all.to_csv(csv_path, index=False)
                print(f"[STEG 6] LIKERT_ALL: path={csv_path}, shape={likert_all.shape}, cols={list(likert_all.columns)}")
                register_output(step="STEG 6", label="LIKERT_ALL_ID", path=csv_path, kind="csv", df=likert_all, note="xlsx_fail")
                for bg, tbl in likert_tables.items():
                    csv_path = _bycat_path.with_name(_bycat_path.stem + f"__LIKERT_{bg}_ID.csv")
                    tbl.to_csv(csv_path, index=False)
                    print(f"[STEG 6] LIKERT_{bg}: path={csv_path}, shape={tbl.shape}, cols={list(tbl.columns)}")
                    register_output(step="STEG 6", label=f"LIKERT_{bg}_ID", path=csv_path, kind="csv", df=tbl, note="xlsx_fail")
    
    # 4) Konsoll-oversikt
    print("\n[OCAI] Bakgrunnsvariabler behandlet:", ", ".join(sorted(list(ocai_tables.keys()))))
    print("[STRATEGI] Bakgrunnsvariabler behandlet:", ", ".join(sorted(list(strat_tables.keys()))))
    for name, tables in [("OCAI", ocai_tables), ("STRATEGI", strat_tables)]:
        for bg, tbl in tables.items():
            _log_df_meta(f"[{name}] {bg} (meta)", tbl)

    # =======================
    # STEG 6B: Master-deskriptiver (OCAI/Strategi Aitchison, Likert, Bakgrunn)
    # =======================
    # - Bruker ID-intersection per blokk (OCAI/Strategi) på tvers av 3 ark
    # - Aitchison-agg per ID via eksisterende _aggregate_profiles_aitchison_per_id
    # - Center (geom. mean + closure) og SD i CLR-rom
    # - Likert og bakgrunn: rå deskriptiver

    def _id_intersection_for_block(block: str, sheets: list[str], p_tables: dict) -> tuple[set[str], list[str]]:
        if not sheets:
            return set(), []
        ids = []
        used_sheets = []
        for sh in sheets:
            if sh not in p_tables:
                continue
            df = p_tables[sh]
            if "ID" not in df.columns:
                continue
            used_sheets.append(sh)
            ids.append(set(df["ID"].astype(str)))
        if not ids:
            return set(), []
        inter = set.intersection(*ids)
        return inter, used_sheets

    def _aitchison_center_sd(P: pd.DataFrame, cols: list[str], eps: float = 1e-12) -> pd.DataFrame:
        """Returnerer Aitchison center (0..1 + pct) og SD i CLR-rom per komponent."""
        if P.empty:
            return pd.DataFrame()
        X = P[cols].to_numpy(float)
        X = np.clip(X, eps, None)
        X = X / X.sum(axis=1, keepdims=True)
        L = np.log(X)
        clr = L - L.mean(axis=1, keepdims=True)
        clr_mean = clr.mean(axis=0)
        clr_sd = clr.std(axis=0, ddof=1)
        center = np.exp(clr_mean)
        center = center / center.sum()
        out = pd.DataFrame({
            "Profil": cols,
            "Mean": center,
            "Mean_pct": center * 100.0,
            "SD_clr": clr_sd,
            "N": int(X.shape[0]),
        })
        return out

    def _likert_desc(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        num_cols = [c for c in df.columns if c != "ID"]
        rows = []
        for c in num_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            n = int(s.notna().sum())
            miss = int(s.isna().sum())
            rows.append({
                "Variabel": c,
                "Mean": float(s.mean()) if n else np.nan,
                "Std": float(s.std(ddof=1)) if n > 1 else np.nan,
                "N": n,
                "Missing": miss,
            })
        return pd.DataFrame(rows)

    def _categorical_desc(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        rows = []
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c].astype("string")
            n = int(s.notna().sum())
            miss = int(s.isna().sum())
            counts = s.value_counts(dropna=True)
            for cat, cnt in counts.items():
                pct = (float(cnt) / n * 100.0) if n else np.nan
                rows.append({
                    "Variabel": c,
                    "Kategori": str(cat),
                    "Count": int(cnt),
                    "Percent": pct,
                    "N": n,
                    "Missing": miss,
                })
        return pd.DataFrame(rows)

    # OCAI / Strategi: ID-intersection + Aitchison
    ocai_ids, ocai_sheets = _id_intersection_for_block("OCAI", _block_sheets.get("OCAI", []), p_tables)
    _strat_key = next((k for k in ["Strategi","STRATEGI","STRAT","Strategy","STRATEGY"] if k in _block_sheets), None)
    strat_ids, strat_sheets = _id_intersection_for_block("Strategi", _block_sheets.get(_strat_key, []) if _strat_key else [], p_tables)

    ocai_stack = []
    for sh in ocai_sheets:
        df = p_tables.get(sh, pd.DataFrame()).copy()
        if not df.empty and "ID" in df.columns:
            df = df[df["ID"].astype(str).isin(ocai_ids)]
            ocai_stack.append(df[["ID"] + ocai_cols])
    strat_stack = []
    for sh in strat_sheets:
        df = p_tables.get(sh, pd.DataFrame()).copy()
        if not df.empty and "ID" in df.columns:
            df = df[df["ID"].astype(str).isin(strat_ids)]
            strat_stack.append(df[["ID"] + strat_cols])

    ocai_stack_df = pd.concat(ocai_stack, axis=0, ignore_index=True) if ocai_stack else pd.DataFrame()
    strat_stack_df = pd.concat(strat_stack, axis=0, ignore_index=True) if strat_stack else pd.DataFrame()

    ocai_expected = len(ocai_sheets)
    strat_expected = len(strat_sheets)
    if ocai_expected and ocai_expected != 3:
        print(f"[STEG 6B] WARNING: OCAI expected_rows_per_id={ocai_expected} (not 3)")
    if strat_expected and strat_expected != 3:
        print(f"[STEG 6B] WARNING: Strategi expected_rows_per_id={strat_expected} (not 3)")
    ocai_agg_desc = _aggregate_profiles_aitchison_per_id(
        ocai_stack_df, ocai_cols, block="OCAI", stage_label="STEG6B/OCAI", expected_rows_per_id=ocai_expected
    ) if not ocai_stack_df.empty else pd.DataFrame()
    strat_agg_desc = _aggregate_profiles_aitchison_per_id(
        strat_stack_df, strat_cols, block="Strategi", stage_label="STEG6B/Strategi", expected_rows_per_id=strat_expected
    ) if not strat_stack_df.empty else pd.DataFrame()

    ocai_all_id_df = _ocai_block.copy() if not _ocai_block.empty else pd.DataFrame()
    strat_all_id_df = _strat_block.copy() if not _strat_block.empty else pd.DataFrame()
    desc_ocai = _aitchison_center_sd(ocai_all_id_df, ocai_cols) if not ocai_all_id_df.empty else pd.DataFrame()
    if desc_ocai.empty and not ocai_agg_desc.empty:
        desc_ocai = _aitchison_center_sd(ocai_agg_desc, ocai_cols)
    desc_strategy = _aitchison_center_sd(strat_all_id_df, strat_cols) if not strat_all_id_df.empty else pd.DataFrame()
    if desc_strategy.empty and not strat_agg_desc.empty:
        desc_strategy = _aitchison_center_sd(strat_agg_desc, strat_cols)

    # Likert (Kontroll)
    sh_ctrl = find_sheet_name("Kontroll", pd.ExcelFile(xlsx_path).sheet_names)
    ctrl_df = pd.read_excel(xlsx_path, sheet_name=sh_ctrl) if sh_ctrl else pd.DataFrame()
    desc_likert = _likert_desc(ctrl_df)

    # Bakgrunn (kategorisk)
    sh_bg = (find_sheet_name("Bakgrunn", pd.ExcelFile(xlsx_path).sheet_names)
             or find_sheet_name("Background", pd.ExcelFile(xlsx_path).sheet_names))
    bg_df = pd.read_excel(xlsx_path, sheet_name=sh_bg) if sh_bg else pd.DataFrame()
    bg_cols = ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]
    desc_bg = _categorical_desc(bg_df, bg_cols)

    # Meta: ID intersection
    desc_meta = pd.DataFrame([
        {"Blokk": "OCAI", "Sheets_used": ", ".join(ocai_sheets), "N_IDs_intersection": int(len(ocai_all_id_df)) if not ocai_all_id_df.empty else int(len(ocai_ids))},
        {"Blokk": "Strategi", "Sheets_used": ", ".join(strat_sheets), "N_IDs_intersection": int(len(strat_all_id_df)) if not strat_all_id_df.empty else int(len(strat_ids))},
    ])

    out_desc = base_path.with_name(base_path.stem + "_descriptives_master.xlsx")
    with pd.ExcelWriter(out_desc, engine="xlsxwriter") as w:
        if not desc_ocai.empty:
            export_excel(desc_ocai, writer=w, sheet_name="desc_ocai", label="desc_ocai")
        if not desc_strategy.empty:
            export_excel(desc_strategy, writer=w, sheet_name="desc_strategy", label="desc_strategy")
        if not desc_likert.empty:
            export_excel(desc_likert, writer=w, sheet_name="desc_likert", label="desc_likert")
        if not desc_bg.empty:
            export_excel(desc_bg, writer=w, sheet_name="desc_background_categorical", label="desc_background_categorical")
        export_excel(desc_meta, writer=w, sheet_name="desc_meta", label="desc_meta")
    print(f"[STEG 6B] Skrev master-deskriptiver til: {out_desc}")
    register_output(step="STEG 6B", label="descriptives_master", path=out_desc, kind="xlsx")

    # Rydder p-matriser, estimerer Dirichlet-alpha per ark/blokk (Minka) og tester mot symmetrisk referanse.
    # Tar inn p_tables/ocai_cols/strat_cols og skriver alpha- og fit-tabeller til <dataset>_dirichlet_alpha.xlsx.
    # =======================
    # STEG 7: Dirichlet-MLE (Minka) med PyPI-pakken `dirichlet`
    # =======================
    # - Rens rader (0..1, sum≈1, ingen 0) for hvert ark og blokk (OCAI/Strategi)
    # - Estimer alpha med Minka (fixedpoint → meanprecision fallback)
    # - Sammenlign mot symmetrisk referanse (MLE for ett felles a) m/ LR-test
    # - Regn pseudo-R², R2_LR, AIC/BIC/AICc
    # - Lagre alpha og fit-statistikk til <eksempeldatasett>_dirichlet_alpha.xlsx
    # - Konsoll-rapport per ark
    # Merk: _chi2, np, pd, p_tables, ocai_cols, strat_cols, xlsx_path forventes i scope
    
    # --- Hjelpere ---
    def _prepare_dirichlet_rows(P: pd.DataFrame, cols, tol=1e-6, eps=1e-8):
        """Rens rader for Dirichlet-MLE:
        - behold kun *cols*
        - dropp rader med NaN
        - behold kun rader som summerer ~1 (±tol)
        - erstatt 0 med eps og re-normaliser (Minka krever > 0)
        Returnerer ndarray (n x k).
        """
        X = P[cols].astype(float).copy()
        X = X.dropna(axis=0, how="any")
        if X.empty:
            return np.empty((0, len(cols)))
        s = X.sum(axis=1).to_numpy()
        keep = np.isfinite(s) & (np.abs(s - 1.0) <= tol)
        X = X.loc[keep]
        if X.empty:
            return np.empty((0, len(cols)))
        A = X.to_numpy(dtype=float)
        A[A < 0] = 0.0
        if (A <= 0).any():
            A[A <= 0] = eps
            A = A / A.sum(axis=1, keepdims=True)
        return A
    
    
    def _dirichlet_alpha_minka(A: np.ndarray):
        """Estimer alpha med Minka (dirichlet.mle). Returnerer (alpha, method) eller (None, None)."""
        if A.size == 0 or A.shape[0] < 5:  # svært lite utvalg kan bli ustabilt
            return None, None
        try:
            return _dir_pkg.mle(A, method="fixedpoint"), "fixedpoint"
        except Exception:
            try:
                return _dir_pkg.mle(A, method="meanprecision"), "meanprecision"
            except Exception:
                return None, None
    
    
    def _ll_sum(A: np.ndarray, alpha: np.ndarray):
        """Sum log-likelihood over rader gitt A (n x k) og alpha (k,).
        Robust, eksplisitt formel (uavhengig av broadcasting).
        Forutsetter radsum≈1 og A>0 (sikres i _prepare_dirichlet_rows).
        """
        if A.size == 0 or alpha is None:
            return np.nan
        X = np.asarray(A, float)  # (n, k)
        lnC = gammaln(alpha.sum()) - gammaln(alpha).sum()  # konstant per rad
        return float(((alpha - 1.0) * np.log(X)).sum() + X.shape[0] * lnC)
    
    # --- Symmetrisk-Dirichlet null (MLE for a) ---
    def _ll_symm_only(A: np.ndarray, a: float) -> float:
        """Log-likelihood for symmetrisk Dirichlet med parameter a (>0)."""
        if not np.isfinite(a) or a <= 0:
            return -np.inf
        k = A.shape[1]
        lnC = gammaln(k * a) - k * gammaln(a)
        return float(((a - 1.0) * np.log(A)).sum() + A.shape[0] * lnC)
    
    def _mle_symmetric_dirichlet(A: np.ndarray) -> float:
        """One-parameter MLE for symmetrisk Dirichlet via begrenset skalær optimalisering."""
        if A.size == 0:
            return np.nan
        res = minimize_scalar(lambda a: -_ll_symm_only(A, a),
                              bounds=(1e-6, 1e3), method="bounded")
        return float(res.x) if res.success else np.nan
    
    
    def _estimate_block_for_sheet(sheet_name, P: pd.DataFrame, cols, block_name):
        """Returner:
        - rows_df: per-profil (Blokk, Ark, Profil, alpha, S, N_used)
        - fit_df : per-ark   (Blokk, Ark, N_used, k, S, LL_full, LL_ref, LR, df, p, 
                              pseudo_R2_nll, R2_LR, AIC_full, BIC_full, AICc_full,
                              LR_conclusion, IC_conclusion, method)
        Referanse: symmetrisk Dirichlet (MLE for a), ikke lenger S/k.
        """
        A = _prepare_dirichlet_rows(P, cols)
        k = len(cols)
        if A.size > 0 and A.shape[1] != k:
            raise ValueError(f"Shape mismatch: A har {A.shape[1]} kolonner, forventet k={k}")
    
        alpha, method = _dirichlet_alpha_minka(A)
        if alpha is None:
            rows_df = pd.DataFrame({
                "Blokk": [block_name]*k,
                "Ark": [sheet_name]*k,
                "Profil": cols,
                "alpha": np.nan,
                "S": np.nan,
                "N_used": A.shape[0],
            })
            fit_df = pd.DataFrame([{
                "Blokk": block_name, "Ark": sheet_name, "N_used": A.shape[0], "k": k, "S": np.nan,
                "LL_full": np.nan, "LL_ref": np.nan, "LR": np.nan, "df": k-1, "p": np.nan,
                "pseudo_R2_nll": np.nan, "R2_LR": np.nan,
                "AIC_full": np.nan, "BIC_full": np.nan, "AICc_full": np.nan,
                "LR_conclusion": None, "IC_conclusion": None, "method": None
            }])
            return rows_df, fit_df
    
        S_hat = float(np.sum(alpha))
        rows_df = pd.DataFrame({
            "Blokk": [block_name]*k,
            "Ark": [sheet_name]*k,
            "Profil": cols,
            "alpha": alpha,
            "S": [S_hat]*k,
            "N_used": [A.shape[0]]*k,
        })
    
        # LL (full)
        LL_full = _ll_sum(A, alpha)
    
        # LL (symmetrisk referanse via MLE for a)
        a_hat_sym = _mle_symmetric_dirichlet(A)
        alpha_ref = np.full(k, a_hat_sym, dtype=float)
        LL_ref  = _ll_sum(A, alpha_ref)
    
        # LR-test (Wilks, df = k−1), med klamp for numerisk stabilitet
        if np.isfinite(LL_full) and np.isfinite(LL_ref):
            LR_raw = 2.0 * (LL_full - LL_ref)
            LR = max(LR_raw, 0.0)
            df = k - 1
            p  = 1.0 - _chi2.cdf(LR, df)
            # Pseudo-R^2 (to alternativer):
            pseudo_R2_nll = compute_pseudo_r2(-LL_full, -LL_ref)
            N = A.shape[0]
            R2_LR = 1.0 - np.exp(-LR / N) if (N > 0 and np.isfinite(LR)) else np.nan
        else:
            LR = np.nan; df = k - 1; p = np.nan; pseudo_R2_nll = np.nan; R2_LR = np.nan
            N = A.shape[0]
    
        # AIC/BIC for fullmodell (p = k parametre)
        N_used = A.shape[0]
        AIC_full = compute_aic(LL_full, k)
        BIC_full = compute_bic(LL_full, k, N_used)
        # AICc (små-utvalgskorrigert)
        if np.isfinite(AIC_full) and (N_used > (k + 1)):
            AICc_full = AIC_full + (2.0 * k * (k + 1)) / (N_used - k - 1)
        else:
            AICc_full = np.nan
    
        # Enkle kriterier/konklusjoner
        if np.isfinite(p) and (p < 0.05) and np.isfinite(LR) and (LL_full > LL_ref):
            LR_conclusion = "Asymmetrisk modell forbedrer vs. symmetrisk (Wilks p<0.05)."
        elif np.isfinite(p):
            LR_conclusion = "Ingen klar forbedring vs. symmetrisk (Wilks p≥0.05)."
        else:
            LR_conclusion = "LR/Wilks ikke beregnbar."
    
        if np.isfinite(AIC_full) and np.isfinite(BIC_full):
            if N_used >= 8 * k:
                IC_conclusion = f"Informasjonskriterier stabile (N={N_used} ≥ 8k={8*k}); bruk AIC/BIC relativt (lavere er bedre)."
            elif N_used > (k + 1) and np.isfinite(AICc_full):
                IC_conclusion = f"Liten N relativt til k (N={N_used} < 8k); bruk AICc (lavere er bedre)."
            else:
                IC_conclusion = f"Svært liten N relativt til k (N={N_used} ≤ k+1); AIC/BIC lite informative."
        else:
            IC_conclusion = "AIC/BIC ikke beregnbar."
    
        fit_df = pd.DataFrame([{
            "Blokk": block_name, "Ark": sheet_name, "N_used": N_used, "k": k, "S": S_hat,
            "LL_full": LL_full, "LL_ref": LL_ref, "LR": LR, "df": df, "p": p,
            "pseudo_R2_nll": pseudo_R2_nll, "R2_LR": R2_LR,
            "AIC_full": AIC_full, "BIC_full": BIC_full, "AICc_full": AICc_full,
            "LR_conclusion": LR_conclusion, "IC_conclusion": IC_conclusion,
            "method": method
        }])
        return rows_df, fit_df
    
    
    # --- Estimer per ark (fra p_tables) + MERGED (Aitchison per ID) ---
    _rows, _fits = [], []

    def _ordered_sheet_names(prefixes: list[str], available: list[str]) -> list[str]:
        out = []
        for pref in prefixes:
            sh = find_sheet_name(pref, available)
            if sh and sh not in out:
                out.append(sh)
        return out

    def _merged_block_aitchison(block: str, prefixes: list[str], cols: list[str]) -> pd.DataFrame:
        xls = pd.ExcelFile(xlsx_path)
        frames = []
        for pref in prefixes:
            sh = find_sheet_name(pref, xls.sheet_names)
            if sh is None:
                continue
            df = pd.read_excel(xlsx_path, sheet_name=sh)
            if "ID" not in df.columns or not set(cols).issubset(df.columns):
                continue
            P = df[["ID"] + cols].copy()
            P["ID"] = P["ID"].astype(str)
            for c in cols:
                P[c] = pd.to_numeric(P[c], errors='coerce') / 100.0
            s = P[cols].sum(axis=1)
            keep = np.isfinite(s) & (np.abs(s - 1.0) <= 1e-6)
            P = P.loc[keep].dropna(subset=cols)
            if not P.empty:
                frames.append(P.reset_index(drop=True))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, axis=0, ignore_index=True)
        counts = out.groupby("ID").size()
        keep_ids = counts[counts == len(frames)].index
        out = out[out["ID"].isin(keep_ids)]
        if out.empty:
            return pd.DataFrame()
        return _aggregate_profiles_aitchison_per_id(
            out, cols, block, stage_label="STEG7/MERGED", expected_rows_per_id=len(frames)
        )

    _ocai_prefixes = ["OCAI - dominerende", "OCAI - strategiske", "OCAI - suksess"]
    _strat_prefixes = ["Strategi - dominerende", "Strategi - strategiske", "Strategi - suksess"]
    _available_sheets = list(p_tables_id.keys())
    _ocai_sheets = _ordered_sheet_names(_ocai_prefixes, _available_sheets)
    _strat_sheets = _ordered_sheet_names(_strat_prefixes, _available_sheets)

    _DIRICHLET_SOURCE_MAP = {}
    for i, sh in enumerate(_ocai_sheets, start=1):
        P = p_tables.get(sh)
        if P is None or not set(ocai_cols).issubset(P.columns):
            continue
        r, f = _estimate_block_for_sheet(sh, P, ocai_cols, "OCAI")
        r["source"] = f"SHEET{i}"
        r["sheet_name"] = sh
        f["source"] = f"SHEET{i}"
        f["sheet_name"] = sh
        _rows.append(r); _fits.append(f)
        _DIRICHLET_SOURCE_MAP[("OCAI", sh)] = f"SHEET{i}"
        if not f.empty and not r.empty:
            _n_obs = int(f["N_used"].iloc[0])
            _S_hat = float(f["S"].iloc[0]) if np.isfinite(f["S"].iloc[0]) else np.nan
            _alpha = np.round(r["alpha"].to_numpy(float), 3).tolist()
            print(f"[DIRICHLET FIT] block=OCAI source=SHEET{i} n_obs={_n_obs} alpha={_alpha} S={_S_hat:.3f}")

    for i, sh in enumerate(_strat_sheets, start=1):
        P = p_tables.get(sh)
        if P is None or not set(strat_cols).issubset(P.columns):
            continue
        r, f = _estimate_block_for_sheet(sh, P, strat_cols, "Strategi")
        r["source"] = f"SHEET{i}"
        r["sheet_name"] = sh
        f["source"] = f"SHEET{i}"
        f["sheet_name"] = sh
        _rows.append(r); _fits.append(f)
        _DIRICHLET_SOURCE_MAP[("Strategi", sh)] = f"SHEET{i}"
        if not f.empty and not r.empty:
            _n_obs = int(f["N_used"].iloc[0])
            _S_hat = float(f["S"].iloc[0]) if np.isfinite(f["S"].iloc[0]) else np.nan
            _alpha = np.round(r["alpha"].to_numpy(float), 3).tolist()
            print(f"[DIRICHLET FIT] block=Strategi source=SHEET{i} n_obs={_n_obs} alpha={_alpha} S={_S_hat:.3f}")

    O_merge = _merged_block_aitchison("OCAI", _ocai_prefixes, ocai_cols)
    if not O_merge.empty:
        r, f = _estimate_block_for_sheet("MERGET_OCAI", O_merge, ocai_cols, "OCAI")
        r["source"] = "MERGED"
        r["sheet_name"] = "MERGED"
        f["source"] = "MERGED"
        f["sheet_name"] = "MERGED"
        _rows.append(r); _fits.append(f)
        _DIRICHLET_SOURCE_MAP[("OCAI", "MERGET_OCAI")] = "MERGED"
        if not f.empty and not r.empty:
            _n_obs = int(f["N_used"].iloc[0])
            _S_hat = float(f["S"].iloc[0]) if np.isfinite(f["S"].iloc[0]) else np.nan
            _alpha = np.round(r["alpha"].to_numpy(float), 3).tolist()
            print(f"[DIRICHLET FIT] block=OCAI source=MERGED n_obs={_n_obs} alpha={_alpha} S={_S_hat:.3f}")

    S_merge = _merged_block_aitchison("Strategi", _strat_prefixes, strat_cols)
    if not S_merge.empty:
        r, f = _estimate_block_for_sheet("MERGET_STRATEGI", S_merge, strat_cols, "Strategi")
        r["source"] = "MERGED"
        r["sheet_name"] = "MERGED"
        f["source"] = "MERGED"
        f["sheet_name"] = "MERGED"
        _rows.append(r); _fits.append(f)
        _DIRICHLET_SOURCE_MAP[("Strategi", "MERGET_STRATEGI")] = "MERGED"
        if not f.empty and not r.empty:
            _n_obs = int(f["N_used"].iloc[0])
            _S_hat = float(f["S"].iloc[0]) if np.isfinite(f["S"].iloc[0]) else np.nan
            _alpha = np.round(r["alpha"].to_numpy(float), 3).tolist()
            print(f"[DIRICHLET FIT] block=Strategi source=MERGED n_obs={_n_obs} alpha={_alpha} S={_S_hat:.3f}")

    DIR_RES  = pd.concat(_rows, axis=0, ignore_index=True) if _rows else pd.DataFrame(
        columns=["Blokk","Ark","Profil","alpha","S","N_used","source","sheet_name"]
    )
    DIR_FITS = pd.concat(_fits, axis=0, ignore_index=True) if _fits else pd.DataFrame(
        columns=["Blokk","Ark","N_used","k","S","LL_full","LL_ref","LR","df","p","pseudo_R2_nll","R2_LR","AIC_full","BIC_full","AICc_full","LR_conclusion","IC_conclusion","method","source","sheet_name"]
    )
    if "block" not in DIR_RES.columns and "Blokk" in DIR_RES.columns:
        DIR_RES.insert(0, "block", DIR_RES["Blokk"])
    if "block" not in DIR_FITS.columns and "Blokk" in DIR_FITS.columns:
        DIR_FITS.insert(0, "block", DIR_FITS["Blokk"])
    _MERGED_PER_ID = {"OCAI": O_merge, "Strategi": S_merge}
    _DIRICHLET_SIM_META = []
    
    # --- Lagre og pen utskrift ---
    _out_dir = xlsx_path.with_name(xlsx_path.stem + "_dirichlet_alpha_ID.xlsx")
    try:
        with pd.ExcelWriter(_out_dir, engine="xlsxwriter") as w:
            export_excel(DIR_RES, writer=w, sheet_name="Dirichlet_alpha", label="Dirichlet_alpha")
            export_excel(DIR_FITS, writer=w, sheet_name="Fit_stats", label="Fit_stats")
    except Exception:
        with pd.ExcelWriter(_out_dir, engine="openpyxl") as w:
            export_excel(DIR_RES, writer=w, sheet_name="Dirichlet_alpha", label="Dirichlet_alpha")
            export_excel(DIR_FITS, writer=w, sheet_name="Fit_stats", label="Fit_stats")
    print(f"Lagret Dirichlet-estimater til: {_out_dir}")
    register_output(step="STEG 7", label="Dirichlet_alpha_ID", path=_out_dir, kind="xlsx")
    print(f"[STEG 7] DIR_RES: path={_out_dir}, shape={DIR_RES.shape}, cols={list(DIR_RES.columns)}")
    _log_df_meta("[STEG 7] DIR_RES (meta)", DIR_RES)
    print(f"[STEG 7] DIR_FITS: path={_out_dir}, shape={DIR_FITS.shape}, cols={list(DIR_FITS.columns)}")
    _log_sig_summary(DIR_FITS, "DIR_FITS", "STEG7")
    _log_dirichlet_fit_rows(DIR_RES, DIR_FITS, cap=200)
    
    for blk in ["OCAI", "Strategi"]:
        sub_rows = DIR_RES[DIR_RES["Blokk"] == blk]
        sub_fit  = DIR_FITS[DIR_FITS["Blokk"] == blk]
        if sub_rows.empty and sub_fit.empty:
            continue
        print(f"\n[{blk}] – per ark: alpha/S/N")
        for ark, d in sub_rows.groupby("Ark", sort=False):
            disp = d[["Profil","alpha","S","N_used"]].copy()
            disp["alpha"] = np.round(disp["alpha"].astype(float), 3)
            disp["S"]     = np.round(disp["S"].astype(float), 3)
            _log_df_meta(f"[{blk}] Ark={ark} (meta)", disp)
        print(f"[{blk}] – fit-statistikk per ark (LL, LR, p, pseudo-R²(NLL), R2_LR, AIC, BIC, AICc, konklusjoner)")
        if not sub_fit.empty:
            dispf = sub_fit.copy()
            for c in ["S","LL_full","LL_ref","LR","p","pseudo_R2_nll","R2_LR","AIC_full","BIC_full","AICc_full"]:
                if c in dispf.columns:
                    dispf[c] = pd.to_numeric(dispf[c], errors="coerce").round(4)
            _log_sig_summary(dispf, f"DIR_FITS_{blk}", "STEG7")
    
    
    # Plotter tetraeder med faktiske p-rader (og eventuelt simulert Dirichlet-sky) for hvert ark og blokk.
    # Input er p_tables og DIR_RES fra steg 7; output er PNG-filer i _dirichlet_plots og ev. plt.show.
    # =======================
    # STEG 8: Visualisering (kun EMPIRISKE data) + Dirichlet-sky
    # =======================
    # Forutsetter i minnet: xlsx_path, p_tables, DIR_RES (fra STEG 7), ocai_cols, strat_cols
    
    # --- 8.1 Geometri: regulært tetraeder og hjelpefunksjoner -------------------
    def _tetra_vertices():
        v1 = np.array([ 1.0,  0.0,  0.0])
        v2 = np.array([-1/3,  2*np.sqrt(2)/3,  0.0])
        v3 = np.array([-1/3, -np.sqrt(2)/3,  np.sqrt(6)/3])
        v4 = np.array([-1/3, -np.sqrt(2)/3, -np.sqrt(6)/3])
        return np.vstack([v1, v2, v3, v4])
    
    _V = _tetra_vertices()
    
    def _bary_to_xyz(P4: np.ndarray) -> np.ndarray:
        """(n,4) med rader ~sum 1 → (n,3) xyz-koordinater i tetra."""
        return P4 @ _V
    
    def _plot_tetra_wire(ax, vertex_labels):
        # kanter
        for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            ax.plot([_V[i,0],_V[j,0]], [_V[i,1],_V[j,1]], [_V[i,2],_V[j,2]], linewidth=1, alpha=0.4)
        # hjørner
        for idx, lab in enumerate(vertex_labels):
            ax.scatter(_V[idx,0], _V[idx,1], _V[idx,2], s=90, marker='^')
            ax.text(_V[idx,0], _V[idx,1], _V[idx,2], '  '+lab, fontsize=9, ha='left', va='bottom')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    
    # --- 8.2 Hent faktiske (empiriske) p-rader brukt i MLE ----------------------
    def _cols_for_block(block):       # navn på de 4 profil-kolonnene
        return ocai_cols if block == "OCAI" else strat_cols
    
    def _safe_name(s: str) -> str:
        return ''.join(ch if ch.isalnum() or ch in ('-_') else '_' for ch in str(s))
    
    def _empirical_matrix_for_sheet(sheet: str, block: str, tol=1e-6, eps=1e-12):
        """
        Returner (n,4) p-matrise for arket 'sheet'.
        For MERGET_* bygges fra p_tables på tvers av relevante ark.
        """
        cols = _cols_for_block(block)
    
        if sheet.startswith("MERGET_"):
            merged = _MERGED_PER_ID.get(block)
            if merged is None or merged.empty or not set(cols).issubset(merged.columns):
                return None
            X = merged[cols].dropna(how="any").to_numpy(float)
        else:
            if sheet not in p_tables or not set(cols).issubset(p_tables[sheet].columns):
                return None
            X = p_tables[sheet][cols].dropna(how="any").to_numpy(float)
    
        if X.size == 0:
            return None
    
        s = X.sum(axis=1)
        keep = np.isfinite(s) & (np.abs(s - 1.0) <= tol)
        X = X[keep]
        if X.size == 0:
            return None
    
        X[X < 0] = 0.0
        zero = (X <= 0)
        if zero.any():
            X[zero] = eps
            X = X / X.sum(axis=1, keepdims=True)
        return X
    
    def _alpha_for_sheet(dir_res_df: pd.DataFrame, block: str, sheet: str):
        """Hent alpha (i riktig kolonnerekkefølge) og S for gitt ark/blokk."""
        cols = _cols_for_block(block)
        sub = dir_res_df[(dir_res_df["Blokk"]==block) & (dir_res_df["Ark"]==sheet)]
        if sub.empty or set(cols) - set(sub["Profil"]):
            return None, None
        sub = sub.set_index("Profil").loc[cols]
        a = sub["alpha"].to_numpy(float)
        if not np.isfinite(a).all():
            return None, None
        return a, float(np.sum(a))
    
    # --- 8.3 Plot per ark: observasjoner + (valgfritt) simulert Dirichlet-sky ---
    SHOW_IN_SPIDER = True     # True = plt.show(); False = bare lagre PNG
    ADD_DIRICHLET  = True     # True = vis også svak Dirichlet-sky (fra estimert alpha)
    
    out_dir = Path(xlsx_path).with_name(Path(xlsx_path).stem + "_dirichlet_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    _rng = np.random.default_rng(12345)  # fast seed for reproduserbarhet
    
    for block in ["OCAI", "Strategi"]:
        sub = DIR_RES[DIR_RES["Blokk"] == block]
        if sub.empty:
            continue
    
        # Plott MERGET_* først, deretter øvrige ark (stabil rekkefølge)
        sheets = sorted(sub["Ark"].unique().tolist(), key=lambda s: (not s.startswith("MERGET_"), s))
    
        for sheet in sheets:
            X = _empirical_matrix_for_sheet(sheet, block)
            a, S_hat = _alpha_for_sheet(DIR_RES, block, sheet)
            if X is None or a is None:
                print(f"[STEG 8] Hopper over {block} / {sheet}: mangler data eller alpha.")
                continue
    
            xyz_emp = _bary_to_xyz(X)
    
            fig = plt.figure(figsize=(5.6, 5.2))
            ax  = fig.add_subplot(111, projection='3d')
            _plot_tetra_wire(ax, _cols_for_block(block))
    
            # (1) Observasjoner
            n_obs = int(xyz_emp.shape[0])
            ax.scatter(xyz_emp[:,0], xyz_emp[:,1], xyz_emp[:,2], s=7, alpha=0.45, marker='o', label=f"Observed (n={n_obs})")

            # (2) Dirichlet-sky (valgfritt)
            _source = _DIRICHLET_SOURCE_MAP.get((block, sheet), "UNKNOWN")
            n_sim = 0
            if ADD_DIRICHLET:
                sim_n   = int(min(2000, max(300, X.shape[0] * 2)))
                print(f"[DIRICHLET SIM] block={block} source={_source} n_sim={sim_n}")
                _DIRICHLET_SIM_META.append({
                    "block": block,
                    "source": _source,
                    "sheet_name": sheet,
                    "n_sim": sim_n,
                    "plot_step": "STEG 8",
                })
                sim     = _rng.dirichlet(a, size=sim_n)
                xyz_sim = _bary_to_xyz(sim)
                n_sim = int(xyz_sim.shape[0])
                ax.scatter(xyz_sim[:,0], xyz_sim[:,1], xyz_sim[:,2], s=5, alpha=0.20, marker='o', label=f"Dirichlet sim (n={n_sim})")
    
            # (3) Marker barysenteret til Dirichlet (α/S)
            bary = (a / S_hat) @ _V
            ax.scatter(bary[0], bary[1], bary[2], s=110, marker='X', linewidths=1.2, zorder=10)
    
            title_text = (f"{block} – {sheet} (source={_source})\n"
                          f"α={np.round(a,3)}  S={S_hat:.3f}  n_obs={n_obs}  n_sim={n_sim}")
            ax.set_title(title_text)
            print(f"[TETRA TITLE] {title_text}")
            if ADD_DIRICHLET:
                ax.legend(loc='best', fontsize=8, framealpha=0.6)
    
            fname = f"{_safe_name(Path(xlsx_path).stem)}__{_safe_name(block)}__{_safe_name(sheet)}.png"
            fpath = out_dir / fname
            plt.tight_layout()
            plt.savefig(fpath, dpi=160, bbox_inches='tight')
            print(f"[TETRA SAVE] {fpath}")
    
            if SHOW_IN_SPIDER:
                plt.show()
            else:
                plt.close(fig)
    
            print(f"[STEG 8] Lagret: {fpath}")
            register_output(step="STEG 8", label="dirichlet_plot", path=fpath, kind="png")
    
    # Kjører ANOVA per departement, Mann–Whitney for leder vs ansatt, og Spearman mot Kontrol A–E via ID-match.
    # Input er p+bg-data og Kontrol-arket; output er sensitivitets-/korrelasjonsstatistikk skrevet til Excel og konsoll.
    # ===========================
    # STEG 9: Sensitiviteter + Spearman mot «Kontrol»-arket (A–E)
    # ===========================
    # Forutsetter i minnet: xlsx_path, base_path, pbg_path, ocai_cols, strat_cols,
    #                       targets, find_sheet_name, _load_merged_block_from_pbg (fra STEG 6)

    # ---- 9.1 Små hjelpere -------------------------------------------------------

    def _ensure_block_merged(block: str) -> pd.DataFrame:
        """Hent MERGED-blokk fra p+bg-filen (Steg 6-funksjonen)."""
        return _load_merged_block_from_pbg(block, ocai_cols, strat_cols)

    def _anova_oneway_per_profile(df_block: pd.DataFrame, profiles: list[str],
                                  group_col: str = "Departement", min_per_grp: int = 2) -> pd.DataFrame:
        """Enveis ANOVA per profil over group_col (sensitivitetsanalyse)."""
        out = []
        if df_block.empty or group_col not in df_block.columns:
            return pd.DataFrame(columns=["Profil","k_grupper","N","F","p"])
        for prof in profiles:
            if prof not in df_block.columns:
                continue
            g = df_block[[group_col, prof]].dropna()
            groups = [v.values for _, v in g.groupby(group_col)[prof] if len(v) >= min_per_grp]
            if len(groups) < 2:
                out.append({"Profil": prof, "k_grupper": len(groups), "N": int(g.shape[0]),
                            "F": np.nan, "p": np.nan})
                continue
            try:
                F, p = f_oneway(*groups)
            except Exception:
                F, p = np.nan, np.nan
            out.append({"Profil": prof, "k_grupper": len(groups), "N": int(g.shape[0]),
                        "F": float(F), "p": float(p)})
        return pd.DataFrame(out)

    def _detect_role_column(cols: list[str]) -> str | None:
        """Finn mulig rolle/leder-kolonne (for MWU)."""
        candidates = ["Stilling", "Rolle", "Leder", "Position", "Tittel"]
        for c in candidates:
            if c in cols:
                return c
        # fuzzy fallback
        for c in cols:
            if re.search(r"leder|rolle|stilling", str(c), flags=re.I):
                return c
        return None

    def _split_leader_employee(series: pd.Series):
        """Del i leder vs ansatt basert på str-innhold."""
        if series.isna().all():
            return None, None
        s = series.astype("string").str.strip().str.lower()
        leader_tokens = {"leder","leiar","leader","manager","sjef"}
        leder_mask = s.apply(lambda x: any(tok in x for tok in leader_tokens) if isinstance(x, str) else False)
        ansatt_mask = ~leder_mask & s.notna()
        if leder_mask.sum()==0 or ansatt_mask.sum()==0:
            return None, None
        return leder_mask, ansatt_mask

    def _mannwhitney_per_profile(df_block: pd.DataFrame, profiles: list[str], min_per_grp: int = 5) -> pd.DataFrame:
        """Mann–Whitney U: leder vs ansatt per profil (sensitivitet)."""
        out = []
        role_col = _detect_role_column(df_block.columns.tolist())
        if df_block.empty or role_col is None:
            return pd.DataFrame(columns=["Profil","n_leder","n_ansatt","U","p"])
        for prof in profiles:
            if prof not in df_block.columns:
                continue
            sub = df_block[[role_col, prof]].dropna()
            if sub.empty:
                out.append({"Profil": prof, "n_leder": 0, "n_ansatt": 0, "U": np.nan, "p": np.nan})
                continue
            lm, am = _split_leader_employee(sub[role_col])
            if lm is None:
                out.append({"Profil": prof, "n_leder": 0, "n_ansatt": 0, "U": np.nan, "p": np.nan})
                continue
            x = sub.loc[lm, prof].astype(float)
            y = sub.loc[am, prof].astype(float)
            if len(x) < min_per_grp or len(y) < min_per_grp:
                out.append({"Profil": prof, "n_leder": int(len(x)), "n_ansatt": int(len(y)), "U": np.nan, "p": np.nan})
                continue
            try:
                U, p = mannwhitneyu(x, y, alternative='two-sided')
            except Exception:
                U, p = np.nan, np.nan
            out.append({"Profil": prof, "n_leder": int(len(x)), "n_ansatt": int(len(y)), "U": float(U), "p": float(p)})
        return pd.DataFrame(out)

    # ---- 9.2 Les «Kontrol»-arket + bygg individ-tabeller (ID) -------------------

    def _load_kontrol_sheet() -> pd.DataFrame:
        """Returner DF med ['ID','A'..'E'] + ev. bakgrunn fra «Kontrol(l)»."""
        xls = pd.ExcelFile(xlsx_path)
        sh = find_sheet_name("Kontrol", xls.sheet_names) or find_sheet_name("Kontroll", xls.sheet_names)
        if sh is None:
            return pd.DataFrame()
        df = pd.read_excel(xlsx_path, sheet_name=sh)
        if "ID" not in df.columns:
            return pd.DataFrame()
        ctrl_cols = [c for c in ["A","B","C","D","E"] if c in df.columns]
        if not ctrl_cols:
            return pd.DataFrame()
        keep_bg = [c for c in ["Departement","Ansiennitet","Alder","Kjønn","Stilling"] if c in df.columns]
        out = df[["ID"] + ctrl_cols + keep_bg].copy()
        out["ID"] = out["ID"].astype(str)
        for c in ctrl_cols:
            out[c] = pd.to_numeric(out[c], errors='coerce')

        return out

    def _merged_profiles_with_id(block: str) -> pd.DataFrame:
        """
        Bygg én tabell per blokk på individnivå (ID + 4 profiler i 0..1).
        Leser *originale* ark fra xlsx_path (IKKE (p)-ark), slik at vi kan join'e på ID.
        Hvis samme ID forekommer i flere ark innen samme blokk, aggregeres med Aitchison-mean (clr-mean).
        """
        xls = pd.ExcelFile(xlsx_path)
        cols = ocai_cols if block == "OCAI" else strat_cols
        frames = []
        for prefix, _ in targets:
            want = (block == "OCAI" and prefix.lower().startswith("ocai")) or (block == "Strategi" and prefix.lower().startswith("strategi"))
            if not want:
                continue
            sh = find_sheet_name(prefix, xls.sheet_names)
            if sh is None:
                continue
            df = pd.read_excel(xlsx_path, sheet_name=sh)
            if "ID" not in df.columns or not set(cols).issubset(df.columns):
                continue
            P = df[["ID"] + cols].copy()
            P["ID"] = P["ID"].astype(str)
            for c in cols:
                P[c] = pd.to_numeric(P[c], errors='coerce') / 100.0
            s = P[cols].sum(axis=1)
            keep = np.isfinite(s) & (np.abs(s - 1.0) <= 1e-6)
            P = P.loc[keep].dropna(subset=cols)
            if not P.empty:
                frames.append(P.reset_index(drop=True))

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, axis=0, ignore_index=True)
        return _aggregate_profiles_aitchison_per_id(out, cols, block,
                                                    stage_label="STEG9/_merged_profiles_with_id")

    def _spearman_profiles_vs_controls(block: str):
        """Spearman mellom blokkens profiler (0..1) og Kontrol A–E via ID-match."""
        profiles = _merged_profiles_with_id(block)
        kontrol   = _load_kontrol_sheet()
        if profiles.empty or kontrol.empty or "ID" not in kontrol.columns:
            return pd.DataFrame(), pd.DataFrame(), 0, []
        ctrl_cols = [c for c in ["A","B","C","D","E"] if c in kontrol.columns]
        if not ctrl_cols:
            return pd.DataFrame(), pd.DataFrame(), 0, []
        M = profiles.merge(kontrol[["ID"] + ctrl_cols], on="ID", how="inner")
        if M.empty:
            return pd.DataFrame(), pd.DataFrame(), 0, []
        prof_cols = ocai_cols if block == "OCAI" else strat_cols
        cols = prof_cols + ctrl_cols
        rho, p = spearmanr(M[cols], axis=0, nan_policy='omit')
        k = len(prof_cols); m = len(ctrl_cols)
        rho_block = pd.DataFrame(rho[:k, k:k+m], index=prof_cols, columns=ctrl_cols)
        p_block   = pd.DataFrame(p[:k,   k:k+m], index=prof_cols, columns=ctrl_cols)
        return rho_block, p_block, len(M), ctrl_cols

    def _spearman_matrix_to_long(rho_df: pd.DataFrame,
                                 p_df: pd.DataFrame | None,
                                 block: str,
                                 space: str,
                                 method: str,
                                 n_used: int | None):
        if rho_df is None or rho_df.empty:
            return pd.DataFrame()
        long = rho_df.stack().reset_index()
        long.columns = ["x_var", "y_var", "rho"]
        if p_df is not None and not p_df.empty:
            p_long = p_df.stack().reset_index()
            p_long.columns = ["x_var", "y_var", "p_value"]
            long = long.merge(p_long, on=["x_var", "y_var"], how="left")
        long.insert(0, "N_used", n_used if n_used is not None else np.nan)
        long.insert(0, "space", space)
        long.insert(0, "method", method)
        long.insert(0, "block", block)
        return long

    # ---- 9.3 ANOVA og MWU (sensitiviteter) -------------------------------------

    _ocai_merged = _ensure_block_merged("OCAI")
    _strat_merged = _ensure_block_merged("Strategi")

    anova_ocai  = _anova_oneway_per_profile(_ocai_merged,  [c for c in ocai_cols if c in _ocai_merged.columns])
    anova_ocai.insert(0, "Blokk", "OCAI")

    anova_strat = _anova_oneway_per_profile(_strat_merged, [c for c in strat_cols if c in _strat_merged.columns])
    anova_strat.insert(0, "Blokk", "Strategi")

    ANOVA_SUMMARY = pd.concat([anova_ocai, anova_strat], axis=0, ignore_index=True)
    print("\n[STEG 9] ANOVA per departement (sensitivitetsanalyse):")
    _log_sig_summary(ANOVA_SUMMARY, "ANOVA_SUMMARY", "STEG9")
    _log_p_allrows(ANOVA_SUMMARY, "STEG9", "ANOVA_SUMMARY_ALL", cap=2000)

    mwu_ocai  = _mannwhitney_per_profile(_ocai_merged,  [c for c in ocai_cols if c in _ocai_merged.columns])
    mwu_ocai.insert(0, "Blokk", "OCAI")

    mwu_strat = _mannwhitney_per_profile(_strat_merged, [c for c in strat_cols if c in _strat_merged.columns])
    mwu_strat.insert(0, "Blokk", "Strategi")

    MWU_SUMMARY = pd.concat([mwu_ocai, mwu_strat], axis=0, ignore_index=True)
    print("\n[STEG 9] Mann–Whitney U (leder vs ansatt) – sensitivitetsanalyse:")
    _log_sig_summary(MWU_SUMMARY, "MWU_SUMMARY", "STEG9")
    _log_p_allrows(MWU_SUMMARY, "STEG9", "MWU_SUMMARY_ALL", cap=2000)

    # ---- 9.4 Spearman: profiler vs Kontrol (A–E) via ID ------------------------

    rho_O_ctrl, p_O_ctrl, N_O_ctrl, used_ctrl_O = _spearman_profiles_vs_controls("OCAI")
    print("\n[STEG 9] Spearman – OCAI-profiler vs Kontrol A–E")
    if N_O_ctrl > 0:
        print(f"N={N_O_ctrl}")
        _log_df_meta("Korrelasjoner (rho) (meta)", rho_O_ctrl)
        _log_df_meta("P-verdier (meta)", p_O_ctrl)
    else:
        print("Ingen matching mellom OCAI-profiler og Kontrol A–E via ID.")

    rho_S_ctrl, p_S_ctrl, N_S_ctrl, used_ctrl_S = _spearman_profiles_vs_controls("Strategi")
    print("\n[STEG 9] Spearman – Strategi-profiler vs Kontrol A–E")
    if N_S_ctrl > 0:
        print(f"N={N_S_ctrl}")
        _log_df_meta("Korrelasjoner (rho) (meta)", rho_S_ctrl)
        _log_df_meta("P-verdier (meta)", p_S_ctrl)
    else:
        print("Ingen matching mellom Strategi-profiler og Kontrol A–E via ID.")

    # ---- 9.5 Eksport til Excel --------------------------------------------------

    _out_stats = base_path.with_name(base_path.stem + "_sensitivitet_og_korrelasjoner.xlsx")
    with pd.ExcelWriter(_out_stats, engine="xlsxwriter") as w:
        anova_export = ANOVA_SUMMARY.copy()
        if "method" not in anova_export.columns:
            anova_export.insert(1, "method", "ANOVA")
        if "space" not in anova_export.columns:
            anova_export.insert(2, "space", "RAW")
        export_excel(anova_export, writer=w, sheet_name="ANOVA_dept", label="ANOVA_dept")
        _log_sig_summary(anova_export, "ANOVA_dept", "STEG9")

        mwu_export = MWU_SUMMARY.copy()
        if "method" not in mwu_export.columns:
            mwu_export.insert(1, "method", "MWU")
        if "space" not in mwu_export.columns:
            mwu_export.insert(2, "space", "RAW")
        export_excel(mwu_export, writer=w, sheet_name="MWU_role", label="MWU_role")
        _log_sig_summary(mwu_export, "MWU_role", "STEG9")
        if N_O_ctrl > 0:
            export_excel(rho_O_ctrl, writer=w, sheet_name="Spearman_OCAI_Kontrol_rho", label="Spearman_OCAI_Kontrol_rho")
            export_excel(p_O_ctrl, writer=w, sheet_name="Spearman_OCAI_Kontrol_p", label="Spearman_OCAI_Kontrol_p")
            long_O_ctrl = _spearman_matrix_to_long(rho_O_ctrl, p_O_ctrl, "OCAI", "RAW", "spearman", N_O_ctrl)
            if not long_O_ctrl.empty:
                export_excel(long_O_ctrl, writer=w, sheet_name="Sp_OCAI_Ktrl_LONG", label="Sp_OCAI_Ktrl_LONG")
                _log_sig_summary(long_O_ctrl, "Sp_OCAI_Ktrl_LONG", "STEG9")
                _log_p_allrows(
                    long_O_ctrl.rename(columns={"p_value": "p"}),
                    "STEG9",
                    "Sp_OCAI_Ktrl_LONG_ALL",
                    cap=2000,
                )
        if N_S_ctrl > 0:
            export_excel(rho_S_ctrl, writer=w, sheet_name="Spearman_Strat_Kontrol_rho", label="Spearman_Strat_Kontrol_rho")
            export_excel(p_S_ctrl, writer=w, sheet_name="Spearman_Strat_Kontrol_p", label="Spearman_Strat_Kontrol_p")
            long_S_ctrl = _spearman_matrix_to_long(rho_S_ctrl, p_S_ctrl, "Strategi", "RAW", "spearman", N_S_ctrl)
            if not long_S_ctrl.empty:
                export_excel(long_S_ctrl, writer=w, sheet_name="Sp_Strat_Ktrl_LONG", label="Sp_Strat_Ktrl_LONG")
                _log_sig_summary(long_S_ctrl, "Sp_Strat_Ktrl_LONG", "STEG9")
                _log_p_allrows(
                    long_S_ctrl.rename(columns={"p_value": "p"}),
                    "STEG9",
                    "Sp_Strat_Ktrl_LONG_ALL",
                    cap=2000,
                )

        run_meta_rows = [
            {
                "outfile": _out_stats.name,
                "sheet_name": "ANOVA_dept",
                "block": "OCAI/Strategi",
                "space": "RAW",
                "method": "ANOVA",
                "variables": "profiles by Departement",
                "n_used": "",
            },
            {
                "outfile": _out_stats.name,
                "sheet_name": "MWU_role",
                "block": "OCAI/Strategi",
                "space": "RAW",
                "method": "MWU",
                "variables": "profiles: Leder vs Ansatt",
                "n_used": "",
            },
        ]
        if N_O_ctrl > 0:
            run_meta_rows.extend([
                {
                    "outfile": _out_stats.name,
                    "sheet_name": "Spearman_OCAI_Kontrol_rho",
                    "block": "OCAI",
                    "space": "RAW",
                    "method": "Spearman",
                    "variables": "OCAI profiles vs Kontroll A-E",
                    "n_used": N_O_ctrl,
                },
                {
                    "outfile": _out_stats.name,
                    "sheet_name": "Spearman_OCAI_Kontrol_p",
                    "block": "OCAI",
                    "space": "RAW",
                    "method": "Spearman",
                    "variables": "OCAI profiles vs Kontroll A-E (p)",
                    "n_used": N_O_ctrl,
                },
                {
                    "outfile": _out_stats.name,
                    "sheet_name": "Sp_OCAI_Ktrl_LONG",
                    "block": "OCAI",
                    "space": "RAW",
                    "method": "Spearman",
                    "variables": "OCAI profiles vs Kontroll A-E (long)",
                    "n_used": N_O_ctrl,
                },
            ])
        if N_S_ctrl > 0:
            run_meta_rows.extend([
                {
                    "outfile": _out_stats.name,
                    "sheet_name": "Spearman_Strat_Kontrol_rho",
                    "block": "Strategi",
                    "space": "RAW",
                    "method": "Spearman",
                    "variables": "Strategi profiles vs Kontroll A-E",
                    "n_used": N_S_ctrl,
                },
                {
                    "outfile": _out_stats.name,
                    "sheet_name": "Spearman_Strat_Kontrol_p",
                    "block": "Strategi",
                    "space": "RAW",
                    "method": "Spearman",
                    "variables": "Strategi profiles vs Kontroll A-E (p)",
                    "n_used": N_S_ctrl,
                },
                {
                    "outfile": _out_stats.name,
                    "sheet_name": "Sp_Strat_Ktrl_LONG",
                    "block": "Strategi",
                    "space": "RAW",
                    "method": "Spearman",
                    "variables": "Strategi profiles vs Kontroll A-E (long)",
                    "n_used": N_S_ctrl,
                },
            ])
        export_excel(pd.DataFrame(run_meta_rows), writer=w, sheet_name="RUN_META", label="RUN_META")

    print(f"\n[STEG 9] Lagret sensitivitets-/korrelasjonsfiler til: {_out_stats}")
    register_output(step="STEG 9", label="sensitivitet_og_korrelasjoner", path=_out_stats, kind="xlsx")

    
    # Kobler ILR-transformerte OCAI/Strategi-profiler med kontroller via ID, kjører robuste OLS (HC3) begge veier og lager predikert andelsmatrise.
    # Input er profilark + Kontrol med ID, output er koeffisienttabeller/designmatriser og fitted andeler i <dataset>_ilr_regresjoner.xlsx.
    # ============================
    # STEG 10: ILR-regresjoner (robuste, HC3)
    # ============================
    # Forutsetter i minnet: xlsx_path, base_path, targets, ocai_cols, strat_cols, find_sheet_name
    #                       (+ funksjonene fra tidligere steg dersom scriptet kjøres samlet)

    # ---- 10.1: ILR-verktøy for k=4 (pivot-basis) --------------------------------
    _PSI_4_PIVOT = np.array([
        [ np.sqrt(3/4),         0.0,                 0.0               ],  # p1 vs (p2,p3,p4)
        [-1/np.sqrt(12),  np.sqrt(2/3),              0.0               ],  # p2 vs (p3,p4)
        [-1/np.sqrt(12), -1/np.sqrt(6),        1/np.sqrt(2)           ],  # p3 vs p4
        [-1/np.sqrt(12), -1/np.sqrt(6),       -1/np.sqrt(2)           ],
    ])

    def _closure10(P):
        """Sikrer strictly positive rader og renormaliserer til sum 1 for ILR-transformasjonen i steg 10."""
        P = np.asarray(P, float)
        P[P <= 0] = 1e-12
        return P / P.sum(axis=1, keepdims=True)

    def _ilr4(P, psi=_PSI_4_PIVOT):
        """Pivot-ILR for k=4 (p1 vs rest, p2 vs {p3,p4}, p3 vs p4); input P (n,4) sum=1, output Z (n,3)."""
        P = _closure10(P)
        L = np.log(P)
        clr = L - L.mean(axis=1, keepdims=True)
        return clr @ psi   # (n,3)

    def _ilr4_inv(Z, psi=_PSI_4_PIVOT):
        """Inverterer pivot-ILR for k=4 tilbake til andeler (n,4) der radene summerer til 1."""
        clr = np.asarray(Z) @ psi.T
        U = np.exp(clr)
        return U / U.sum(axis=1, keepdims=True)

    def _psi_pivot_isolate_first(first_idx=0):
        """Returnerer pivot-Ψ-matrise der valgt indeks isoleres som første komponent (for ilr4/ilr4_inv)."""
        if first_idx == 0:
            return _PSI_4_PIVOT
        order = np.r_[first_idx, np.delete(np.arange(4), first_idx)]
        return _PSI_4_PIVOT[order, :]

    # ---- 10.2: Datainnhenting for ID-kobling ------------------------------------
    def _read_bakgrunn(xlsx_path: Path) -> pd.DataFrame:
        """Leser 'Bakgrunn' (ID + 5 BG-kolonner)."""
        xls = pd.ExcelFile(xlsx_path)
        sh = next((s for s in xls.sheet_names if s.lower().startswith("bakgrunn")), None)
        if sh is None:
            return pd.DataFrame()
        df = pd.read_excel(xlsx_path, sheet_name=sh)
        cols_keep = ["ID","Departement","Ansiennitet","Alder","Kjønn","Stilling"]
        have = [c for c in cols_keep if c in df.columns]
        if "ID" not in have:
            return pd.DataFrame()
        out = df[have].copy()
        out["ID"] = out["ID"].astype(str)
        return out

    def _read_kontrol(xlsx_path: Path) -> pd.DataFrame:
        """Leser Kontrol(l)-ark med ID og Likert A–E (kun)."""
        xls = pd.ExcelFile(xlsx_path)
        sh = next((s for s in xls.sheet_names if s.lower().startswith("kontrol")), None)
        if sh is None:
            return pd.DataFrame()
        df = pd.read_excel(xlsx_path, sheet_name=sh)
        if "ID" not in df.columns:
            return pd.DataFrame()
        keep = ["ID"] + [c for c in ["A","B","C","D","E"] if c in df.columns]
        out = df[keep].copy()
        out["ID"] = out["ID"].astype(str)
        for c in ["A","B","C","D","E"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        return out

    def _read_profiles_with_id(block: str, xlsx_path: Path, targets, ocai_cols, strat_cols) -> pd.DataFrame:
        """
        Henter profilark for valgt blokk med ID, skalerer til 0..1, filtrerer sum≈1
        og aggregerer per ID med Aitchison-mean (clr-mean).
        """
        xls = pd.ExcelFile(xlsx_path)
        cols = ocai_cols if block == "OCAI" else strat_cols
        frames = []
        for prefix, _ in targets:
            want = (block == "OCAI" and prefix.lower().startswith("ocai")) or (block == "Strategi" and prefix.lower().startswith("strategi"))
            if not want:
                continue
            sh = next((s for s in xls.sheet_names if s.lower().startswith(prefix.lower())), None)
            if sh is None:
                continue
            df = pd.read_excel(xlsx_path, sheet_name=sh)
            if "ID" not in df.columns or not set(cols).issubset(df.columns):
                continue
            P = df[["ID"] + cols].copy()
            P["ID"] = P["ID"].astype(str)
            for c in cols:
                P[c] = pd.to_numeric(P[c], errors="coerce") / 100.0
            s = P[cols].sum(axis=1)
            keep = np.isfinite(s) & (np.abs(s - 1.0) <= 1e-6)
            P = P.loc[keep].dropna(subset=cols)
            if not P.empty:
                frames.append(P.reset_index(drop=True))

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, axis=0, ignore_index=True)
        return _aggregate_profiles_aitchison_per_id(out, cols, block,
                                                    stage_label="STEG10/_read_profiles_with_id")

    # ---- 10.3: Designmatriser (full-rank dummies) -------------------------------
    def _one_hot_fullrank(df_cats: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Lager full-rank one-hot-dummies for oppgitte kategorikolonner (drop_first) gitt et DF med kategorier."""
        if df_cats is None or df_cats.empty or not cols:
            return pd.DataFrame(index=(df_cats.index if df_cats is not None else None))
        out = []
        for c in cols:
            if c in df_cats.columns:
                s = df_cats[c].astype("string")
                d = pd.get_dummies(s, prefix=c, dtype=float, drop_first=True)
                out.append(d)
        return pd.concat(out, axis=1) if out else pd.DataFrame(index=df_cats.index)

    # ---- 10.4: Bygg analyse-tabell (ID-merge) -----------------------------------
    _S_id = _read_profiles_with_id("Strategi", xlsx_path, targets, ocai_cols, strat_cols)
    _O_id = _read_profiles_with_id("OCAI",    xlsx_path, targets, ocai_cols, strat_cols)
    _K_ctrl = _read_kontrol(xlsx_path)
    _K_bg   = _read_bakgrunn(xlsx_path)

    print(f"[STEG 10] Ark i Excel: {pd.ExcelFile(xlsx_path).sheet_names}")

    if _S_id.empty or _O_id.empty or (_K_ctrl.empty and _K_bg.empty):
        print("[STEG 10] Mangler data for ID-merge (Strategi/OCAI/Kontrol/Bakgrunn). Hopper over regresjoner.")
    else:
        print(f"[STEG 10] ID dtype S={_S_id['ID'].dtype}, O={_O_id['ID'].dtype}, K_ctrl={_K_ctrl['ID'].dtype if not _K_ctrl.empty else 'None'}, K_bg={_K_bg['ID'].dtype if not _K_bg.empty else 'None'}")

        def _merge_with_log(left, right, label_left, label_right):
            if right.empty:
                print(f"[STEG 10] {label_right} er tom; hopper over merge.")
                return left
            n_before = len(left)
            out = left.merge(right, on="ID", how="inner")
            print(f"[STEG 10] Merge {label_left} x {label_right}: {n_before} -> {len(out)} (inner)")
            return out

        D = _merge_with_log(_S_id, _O_id, "S_id", "O_id")
        D = _merge_with_log(D, _K_bg, "S_O", "Bakgrunn")
        D = _merge_with_log(D, _K_ctrl, "S_O_BG", "Kontrol")

        if D.empty:
            print("[STEG 10] Ingen overlappende ID-er etter merges. Hopper over regresjoner.")
        else:
            # skaler sikkerhet
            if D[strat_cols].max().max() > 1.0: D[strat_cols] = D[strat_cols] / 100.0
            if D[ocai_cols].max().max()   > 1.0: D[ocai_cols]   = D[ocai_cols]   / 100.0

            def _oksum(cols):
                s = D[cols].sum(axis=1)
                return np.isfinite(s) & (np.abs(s-1.0) <= 1e-6)

            D = D.loc[_oksum(strat_cols) & _oksum(ocai_cols)].reset_index(drop=True)
            if D.empty:
                print("[STEG 10] Etter sum≈1-filter er det ingen rader igjen.")
            else:
                if D["ID"].duplicated().any():
                    dup_ids = D["ID"][D["ID"].duplicated()].unique().tolist()
                    raise RuntimeError(
                        f"[STEG 10] Dupliserte ID-er etter merge "
                        f"(n={len(dup_ids)}, sample={_redact_list(dup_ids[:20])})"
                    )
                print(f"[STEG 10] ID uniqueness: rows={len(D)}, unique={D['ID'].nunique()}, has_dupes={D['ID'].duplicated().any()}")
                print(f"[STEG 10] Modell-datasett etter merge/filter: shape={D.shape}")
                print(f"[STEG 10] Kolonner (sortert): {sorted(D.columns.tolist())}")
                _log_df("[STEG 10] ID head:", D[["ID"]], preview_rows=5, index=False)

                likert_cols = [c for c in ["A","B","C","D","E"] if c in D.columns]
                X_ctrl = D[likert_cols].apply(pd.to_numeric, errors="coerce") if likert_cols else pd.DataFrame(index=D.index)
                for c in likert_cols:
                    col = X_ctrl[c]
                    na_share = col.isna().mean()
                    col_min = col.min(skipna=True)
                    col_max = col.max(skipna=True)
                    print(f"[STEG 10] Likert {c}: NaN-andel={na_share:.3f}, min={col_min}, max={col_max}")
                print(f"[STEG 10] X_ctrl shape={X_ctrl.shape}, cols={list(X_ctrl.columns)}")

                def _bg_design_with_logging(df_bg: pd.DataFrame, cols: list) -> pd.DataFrame:
                    if df_bg is None or df_bg.empty or not cols:
                        return pd.DataFrame(index=(df_bg.index if df_bg is not None else None))
                    parts = []
                    for c in cols:
                        if c not in df_bg.columns:
                            continue
                        s = df_bg[c]
                        nuniq = s.nunique(dropna=True)
                        nnan = s.isna().sum()
                        num = pd.to_numeric(s, errors="coerce")
                        frac_num = num.notna().mean() if len(num) else 0.0
                        if frac_num >= 0.8:
                            design = num.to_frame(name=c)
                            kind = "numeric"
                            ndummies = 1
                        else:
                            used = s.astype("string")
                            design = pd.get_dummies(used, prefix=c, dtype=float, drop_first=True)
                            kind = "categorical"
                            ndummies = design.shape[1]
                        print(f"[STEG 10] BG {c}: uniq={nuniq}, NaN={nnan}, frac_numeric={frac_num:.2f}, "
                              f"treated_as={kind}, dummy_cols={ndummies}")
                        if not design.empty:
                            parts.append(design)
                    if parts:
                        return pd.concat(parts, axis=1)
                    return pd.DataFrame(index=df_bg.index)

                bg_cols = [c for c in ["Departement","Ansiennitet","Alder","Kjønn","Stilling"] if c in D.columns]
                D_bg_raw = D[bg_cols].copy() if bg_cols else pd.DataFrame(index=D.index)
                X_bg = _bg_design_with_logging(D_bg_raw, bg_cols)
                if bg_cols and X_bg.empty:
                    raise RuntimeError("[STEG 10] BG finnes i data men X_bg ble tom. Sjekk kodingen.")
                print(f"[STEG 10] X_bg shape={X_bg.shape}, cols_example={list(X_bg.columns[:10])}")

                PSI_O = _psi_pivot_isolate_first(first_idx=0)
                Z_O   = pd.DataFrame(_ilr4(D[ocai_cols].to_numpy(), psi=PSI_O),
                                     columns=[f"ilrO_{i+1}" for i in range(3)], index=D.index)

                PSI_S = _psi_pivot_isolate_first(first_idx=0)
                Z_S   = pd.DataFrame(_ilr4(D[strat_cols].to_numpy(), psi=PSI_S),
                                     columns=[f"ilrS_{i+1}" for i in range(3)], index=D.index)
                print(f"[STEG 10] ILR-pred (OCAI) Z_O shape={Z_O.shape}, cols={list(Z_O.columns)}")
                print(f"[STEG 10] ILR-resp (Strategi) Z_S shape={Z_S.shape}, cols={list(Z_S.columns)}")

                def _fit_variant(Z_resp: pd.DataFrame, X_pred: pd.DataFrame, variant_label: str):
                    models = {}
                    X = sm.add_constant(X_pred, has_constant='add')
                    for col in Z_resp.columns:
                        res = sm.OLS(Z_resp[col].to_numpy(), X.to_numpy()).fit(cov_type="HC3")
                        if len(res.params) != X.shape[1]:
                            raise RuntimeError(f"[STEG 10] Param/kolonne mismatch i {variant_label} ({col}): "
                                               f"{len(res.params)} vs {X.shape[1]}")
                        res._design_cols = X.columns.tolist()
                        model_family, variant = _split_model_label(variant_label)
                        y_block, x_block = _infer_blocks_from_family(model_family)
                        model_id = _make_model_id("STEG10", model_family, variant, col)
                        stats = _fit_stats_dict(res)
                        _log_modeldef(
                            "STEG10",
                            model_id,
                            model_family,
                            variant,
                            col,
                            y_block,
                            x_block,
                            "HC3",
                            stats,
                            res._design_cols,
                        )
                        _log_fitrow("STEG10", model_id, model_family, variant, col, "HC3", stats)
                        _log_coefrows("STEG10", model_id, model_family, variant, col, res._design_cols, res, "HC3", stats)
                        models[col] = res
                    print(f"[STEG 10] Variant {variant_label}: X shape={X.shape}, cols={list(X.columns)}")
                    return models, X

                variants = []
                variants.append(("Strategy ~ OCAI | ILR only", Z_S, Z_O))
                if not X_bg.empty:
                    variants.append(("Strategy ~ OCAI | ILR+BG", Z_S, pd.concat([Z_O, X_bg], axis=1)))
                if not X_ctrl.empty:
                    variants.append(("Strategy ~ OCAI | ILR+Likert", Z_S, pd.concat([Z_O, X_ctrl], axis=1)))
                if (not X_bg.empty) and (not X_ctrl.empty):
                    variants.append(("Strategy ~ OCAI | ILR+BG+Likert", Z_S, pd.concat([Z_O, X_bg, X_ctrl], axis=1)))

                variants_rev = []
                variants_rev.append(("OCAI ~ Strategy | ILR only", Z_O, Z_S))
                if not X_bg.empty:
                    variants_rev.append(("OCAI ~ Strategy | ILR+BG", Z_O, pd.concat([Z_S, X_bg], axis=1)))
                if not X_ctrl.empty:
                    variants_rev.append(("OCAI ~ Strategy | ILR+Likert", Z_O, pd.concat([Z_S, X_ctrl], axis=1)))
                if (not X_bg.empty) and (not X_ctrl.empty):
                    variants_rev.append(("OCAI ~ Strategy | ILR+BG+Likert", Z_O, pd.concat([Z_S, X_bg, X_ctrl], axis=1)))

                all_models_S = {}
                all_design_S = {}
                for label, Z_resp, X_pred in variants:
                    m, Xd = _fit_variant(Z_resp, X_pred, label)
                    all_models_S[label] = m
                    all_design_S[label] = Xd

                all_models_O = {}
                all_design_O = {}
                for label, Z_resp, X_pred in variants_rev:
                    m, Xd = _fit_variant(Z_resp, X_pred, label)
                    all_models_O[label] = m
                    all_design_O[label] = Xd

                # Valg av rikeste variant som hovedmodell for senere steg
                pref_S = ["Strategy ~ OCAI | ILR+BG+Likert", "Strategy ~ OCAI | ILR+BG", "Strategy ~ OCAI | ILR+Likert", "Strategy ~ OCAI | ILR only"]
                pref_O = ["OCAI ~ Strategy | ILR+BG+Likert", "OCAI ~ Strategy | ILR+BG", "OCAI ~ Strategy | ILR+Likert", "OCAI ~ Strategy | ILR only"]
                chosen_S = next((p for p in pref_S if p in all_models_S), None)
                chosen_O = next((p for p in pref_O if p in all_models_O), None)
                models_S = all_models_S.get(chosen_S, {})
                models_O = all_models_O.get(chosen_O, {})
                X_all = all_design_S.get(chosen_S, pd.DataFrame(index=D.index))
                X_rev = all_design_O.get(chosen_O, pd.DataFrame(index=D.index))
                print(f"[STEG 10] Valgt hovedvariant (Strategy ~ OCAI): {chosen_S}")
                print(f"[STEG 10] Valgt hovedvariant (OCAI ~ Strategy): {chosen_O}")

                def _coef_table_variant(models_dict, label):
                    rows = []
                    for mdl_label, mdict in models_dict.items():
                        for resp, m in mdict.items():
                            cols = getattr(m, "_design_cols", m.model.exog_names)
                            m_rob, cov, se, tvals, pvals, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
                                m, prefer="HC3", fallback="HC1"
                            )
                            if not cov_stable:
                                params = np.asarray(m.params)
                                se = np.full_like(params, np.nan, dtype=float)
                                tvals = np.full_like(params, np.nan, dtype=float)
                                pvals = np.full_like(params, np.nan, dtype=float)
                            cov_note = "ok" if cov_stable else ";".join(cov_reasons) if cov_reasons else "HC3_invalid"
                            rows.append(pd.DataFrame({
                                "variant": mdl_label,
                                "model": label,
                                "response": resp,
                                "term": cols,
                                "coef": m.params,
                                "se_HC3": se,
                                "t": tvals,
                                "p": pvals,
                                "R2": m.rsquared,
                                "robust_cov_prefer": "HC3",
                                "robust_cov_used": cov_type_used,
                                "robust_cov_note": cov_note,
                                "robust_cov_finite": cov_stable,
                            }))
                    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()

                COEF_S = _coef_table_variant(all_models_S, "Strategy ~ OCAI")
                COEF_O = _coef_table_variant(all_models_O, "OCAI ~ Strategy")

                # Prediksjoner (valgte hovedvarianter brukes hvis tilgjengelig)
                if models_S and not X_all.empty:
                    ZS_hat = np.column_stack([models_S[c].predict(X_all.to_numpy()) for c in models_S])
                    P_S_hat = _ilr4_inv(ZS_hat, psi=PSI_S)
                    P_S_hat_df = pd.DataFrame(P_S_hat, columns=[f"pred_{c}" for c in strat_cols], index=D.index)
                else:
                    P_S_hat_df = pd.DataFrame()

                if models_O and not X_rev.empty:
                    ZO_hat = np.column_stack([models_O[c].predict(X_rev.to_numpy()) for c in models_O])
                    P_O_hat = _ilr4_inv(ZO_hat, psi=PSI_O)
                    P_O_hat_df = pd.DataFrame(P_O_hat, columns=[f"pred_{c}" for c in ocai_cols], index=D.index)
                else:
                    P_O_hat_df = pd.DataFrame()

                out_ilr_reg = base_path.with_name(base_path.stem + "_ilr_regresjoner.xlsx")
                with pd.ExcelWriter(out_ilr_reg, engine="xlsxwriter") as w:
                    if not COEF_S.empty:
                        export_excel(COEF_S, writer=w, sheet_name="Strategy_on_OCAI_all", label="Strategy_on_OCAI_all")
                        _log_sig_summary(COEF_S, "Strategy_on_OCAI_all", "STEG10", cap=500)
                    if not COEF_O.empty:
                        export_excel(COEF_O, writer=w, sheet_name="OCAI_on_Strategy_all", label="OCAI_on_Strategy_all")
                        _log_sig_summary(COEF_O, "OCAI_on_Strategy_all", "STEG10", cap=500)
                    for lbl, Xd in {**all_design_S, **all_design_O}.items():
                        export_excel(pd.DataFrame(Xd), writer=w, sheet_name=lbl[:31], label=lbl[:31])
                    if not P_S_hat_df.empty:
                        export_excel(P_S_hat_df.reset_index(drop=True), writer=w, sheet_name="Strategy_fitted_shares", label="Strategy_fitted_shares")
                    if not P_O_hat_df.empty:
                        export_excel(P_O_hat_df.reset_index(drop=True), writer=w, sheet_name="OCAI_fitted_shares", label="OCAI_fitted_shares")

                print(f"[STEG 10] Skrev ILR-varianter til: {out_ilr_reg}")
                register_output(step="STEG 10", label="ilr_regresjoner", path=out_ilr_reg, kind="xlsx")
                if not COEF_S.empty:
                    print("[STEG 10] Eksempel koeff (Strategy ~ OCAI), inkludert BG/Likert hvis tilgjengelig:")
                    _log_sig_summary(COEF_S, "COEF_S", "STEG10")
                if not COEF_O.empty:
                    print("[STEG 10] Eksempel koeff (OCAI ~ Strategy), inkludert BG/Likert hvis tilgjengelig:")
                    _log_sig_summary(COEF_O, "COEF_O", "STEG10")

    
    # Bygger ILR-balansetabeller fra p+bg, beregner Spearman innen/kryss blokk og (om mulig) MANOVA per bakgrunnsvariabel.
    # Input er pbg-flettede data og ocai/strat-profiler; output er Spearman-/MANOVA-filer på Excel og konsolltabeller.
    # ============================
    # STEG 11: Spearman (ILR) + MANOVA (ILR) — Sensitivitetsanalyse
    # ============================
    # Forutsetter i minnet: base_path, xlsx_path, ocai_cols, strat_cols,
    #                       _load_merged_block_from_pbg (fra STEG 6)
    # Pakker: statsmodels>=0.14 for MANOVA

    # MANOVA er valgfritt (gir melding hvis utilgjengelig)
    try:
        from statsmodels.multivariate.manova import MANOVA
        _HAS_MANOVA_11 = True
    except Exception:
        _HAS_MANOVA_11 = False
        print("[STEG 11] MANOVA ikke tilgjengelig (statsmodels). Spearman kjøres likevel.")

    # ---- 11.1: ILR (pivot, k=4) -------------------------------------------------

    _PSI11 = np.array([
        [ np.sqrt(3/4),         0.0,                 0.0               ],  # p1 vs (p2,p3,p4)
        [-1/np.sqrt(12),  np.sqrt(2/3),              0.0               ],  # p2 vs (p3,p4)
        [-1/np.sqrt(12), -1/np.sqrt(6),        1/np.sqrt(2)           ],  # p3 vs p4
        [-1/np.sqrt(12), -1/np.sqrt(6),       -1/np.sqrt(2)           ],
    ])

    def _closure11(P):
        """Closure-funksjon for steg 11: setter små verdier til eps og normaliserer rader til sum 1."""
        P = np.asarray(P, float)
        P[P <= 0] = 1e-12
        return P / P.sum(axis=1, keepdims=True)

    def _ilr11(P, psi=_PSI11):
        """ILR-transformasjon for 4-dimensjonale profiler i steg 11 med gitt Ψ; returnerer Z (n,3)."""
        P = _closure11(P)
        L = np.log(P)
        clr = L - L.mean(axis=1, keepdims=True)
        return clr @ psi  # (n,3)

    def _pivot_isolate_first11(first_idx=0):
        """Justert Ψ for steg 11 der ønsket profil settes først i pivot-basis."""
        if first_idx == 0:
            return _PSI11
        order = np.r_[first_idx, np.delete(np.arange(4), first_idx)]
        return _PSI11[order, :]

    # ---- 11.2: Hent sammenslåtte blokker fra p+bg (STEG 6) ----------------------

    _ocai_m = _load_merged_block_from_pbg("OCAI", ocai_cols, strat_cols)
    _strat_m = _load_merged_block_from_pbg("Strategi", ocai_cols, strat_cols)

    def _aitchison_mean_by_id(df: pd.DataFrame, cols: list[str], id_col: str = "ID", eps: float = 1e-12) -> pd.DataFrame:
        """Aitchison-mean (clr-mean) per ID for komposisjoner i cols (forutsetter 0..1 og sum≈1 per rad)."""
        tmp = df[[id_col] + cols].copy()
        tmp[id_col] = tmp[id_col].astype(str)

        X = tmp[cols].to_numpy(float)
        X = np.clip(X, eps, None)
        X = X / X.sum(axis=1, keepdims=True)

        logX = np.log(X)
        clr  = logX - logX.mean(axis=1, keepdims=True)

        clr_df = pd.DataFrame(clr, columns=cols)
        clr_df.insert(0, id_col, tmp[id_col].to_numpy())
        clr_mean = clr_df.groupby(id_col)[cols].mean()

        P = np.exp(clr_mean.to_numpy(float))
        P = P / P.sum(axis=1, keepdims=True)

        agg = clr_mean.reset_index()
        agg[cols] = P
        return agg

    def _prep_block11(df, cols):
        """Drop NaN, skaler til 0..1 ved behov, behold rader med sum≈1. Aggreger per ID med Aitchison-mean hvis ID finnes."""
        if df.empty:
            return pd.DataFrame()
        have = [c for c in cols if c in df.columns]
        if len(have) != 4:
            return pd.DataFrame()

        out = df.copy().dropna(subset=have)
        if out[have].max().max() > 1.0:
            out[have] = out[have] / 100.0

        s = out[have].sum(axis=1)
        out = out.loc[np.isfinite(s) & (np.abs(s - 1.0) <= 1e-6)].reset_index(drop=True)

        # Aggreger per respondent (ID) med Aitchison-mean dersom mulig
        if "ID" in out.columns and not out.empty:
            out["ID"] = out["ID"].astype(str)

            meta_cols = [c for c in out.columns if c not in have and c != "ID"]
            if meta_cols:
                meta = out.groupby("ID")[meta_cols].first().reset_index()
            else:
                meta = out[["ID"]].drop_duplicates().reset_index(drop=True)

            comp = _aitchison_mean_by_id(out, have, id_col="ID", eps=1e-12)
            out = meta.merge(comp, on="ID", how="inner")

            # Stabil kolonnerekkefølge
            out = out[["ID"] + meta_cols + have].reset_index(drop=True)

        return out

    _ocai_m  = _prep_block11(_ocai_m,  ocai_cols)
    _strat_m = _prep_block11(_strat_m, strat_cols)

    # ---- 11.3: Lag ILR-koordinater (z1..z3) ------------------------------------

    PSI_O11 = _pivot_isolate_first11(first_idx=0)
    PSI_S11 = _pivot_isolate_first11(first_idx=0)

    def _ilr_df11(df, cols, psi, prefix):
        Z = pd.DataFrame(
            _ilr11(df[cols].to_numpy(), psi=psi),
            index=df.index,
            columns=[f"{prefix}{i+1}" for i in range(3)]
        )
        return pd.concat([df.reset_index(drop=True), Z.reset_index(drop=True)], axis=1)

    _ocai_ilr  = _ocai_m  if _ocai_m.empty  else _ilr_df11(_ocai_m,  ocai_cols,  PSI_O11, "ilrO_")
    _strat_ilr = _strat_m if _strat_m.empty else _ilr_df11(_strat_m, strat_cols, PSI_S11, "ilrS_")

    # ---- 11.4: Spearman (innen blokk og kryss-blokk hvis mulig) ----------------

    def _spearman_matrix11(A: pd.DataFrame, colsA: list, B: pd.DataFrame = None, colsB: list = None):
        """
        Hvis B=None: Spearman rho/p mellom kolonner i A (colsA).
        Hvis B er gitt: Spearman mellom A[colsA] og B[colsB] på matchede rader (primært via ID hvis mulig).
        """
        if A.empty or not colsA:
            return None, None

        if B is None:
            mat = A[colsA].to_numpy()
            rho, p = spearmanr(mat, axis=0, nan_policy='omit')
            k = len(colsA)
            return (
                pd.DataFrame(rho[:k, :k], index=colsA, columns=colsA),
                pd.DataFrame(p[:k, :k],   index=colsA, columns=colsA)
            )

        # Kryss-blokk: match på ID hvis mulig (etter aggregering skal dette være 1 rad per ID)
        if B is None or B.empty or not colsB:
            return None, None

        if ("ID" in A.columns) and ("ID" in B.columns):
            M = A[["ID"] + colsA].merge(B[["ID"] + colsB], on="ID", how="inner")
            if len(M) < 3:
                return None, None
            mat = M[colsA + colsB].to_numpy()
        else:
            # fallback (samme som før): kutt til min(nA, nB)
            n = min(len(A), len(B))
            if n < 3:
                return None, None
            mat = np.column_stack([A[colsA].to_numpy()[:n, :], B[colsB].to_numpy()[:n, :]])

        rho, p = spearmanr(mat, axis=0, nan_policy='omit')
        k, m = len(colsA), len(colsB)
        return (
            pd.DataFrame(rho[:k, k:k+m], index=colsA, columns=colsB),
            pd.DataFrame(p[:k,   k:k+m], index=colsA, columns=colsB)
        )

    _ilrO_cols = [c for c in _ocai_ilr.columns if c.startswith("ilrO_")] if not _ocai_ilr.empty else []
    _ilrS_cols = [c for c in _strat_ilr.columns if c.startswith("ilrS_")] if not _strat_ilr.empty else []

    rho_OO, p_OO = _spearman_matrix11(_ocai_ilr,  _ilrO_cols) if _ilrO_cols else (None, None)
    rho_SS, p_SS = _spearman_matrix11(_strat_ilr, _ilrS_cols) if _ilrS_cols else (None, None)
    rho_OS, p_OS = _spearman_matrix11(_ocai_ilr,  _ilrO_cols, _strat_ilr, _ilrS_cols) if (_ilrO_cols and _ilrS_cols) else (None, None)

    # ---- 11.5: MANOVA på ILR-balansene per bakgrunnsvariabel -------------------

    def _manova_block11(df_ilr: pd.DataFrame, ilr_cols_prefix: str, block_name: str):
        """Returnerer liste med korte rapport-DFer for hver bakgrunnsvariabel (dersom MANOVA finnes)."""
        reports = []
        if not _HAS_MANOVA_11 or df_ilr.empty:
            return reports

        df_ilr = df_ilr.copy()
        zcols = [c for c in df_ilr.columns if c.startswith(ilr_cols_prefix)]
        bg_vars = [c for c in ["Departement","Ansiennitet","Alder","Kjønn","Stilling"] if c in df_ilr.columns]
        for bg in bg_vars:
            df_ilr[bg] = df_ilr[bg].astype("category")

        for bg in bg_vars:
            sub = df_ilr[zcols + [bg]].dropna()
            if sub.empty:
                continue
            # behold bare nivåer med >=2 observasjoner
            counts = sub[bg].astype("string").value_counts()
            keep_levels = counts.index[counts >= 2].tolist()
            sub = sub[sub[bg].astype("string").isin(keep_levels)]
            if sub[bg].nunique() < 2 or len(sub) < 20:
                continue
            try:
                lhs = " + ".join(zcols)
                formula = f"{lhs} ~ C({bg})"
                mv = MANOVA.from_formula(formula, data=sub)
                summ = mv.mv_test()  # dict-lignende; stringifiser
                rep_df = pd.DataFrame({
                    "Block":[block_name],
                    "Background":[bg],
                    "N":[len(sub)],
                    "Levels":[sub[bg].nunique()],
                    "Summary":[str(summ)]
                })
                reports.append(rep_df)
            except Exception as e:
                reports.append(pd.DataFrame({
                    "Block":[block_name],
                    "Background":[bg],
                    "N":[len(sub)],
                    "Levels":[sub[bg].nunique()],
                    "Summary":[f"MANOVA error: {repr(e)}"]
                }))
        return reports

    manova_O_reports = _manova_block11(_ocai_ilr,  "ilrO_", "OCAI")
    manova_S_reports = _manova_block11(_strat_ilr, "ilrS_", "Strategi")

    MANOVA_O = pd.concat(manova_O_reports, axis=0, ignore_index=True) if manova_O_reports else pd.DataFrame(columns=["Block","Background","N","Levels","Summary"])
    MANOVA_S = pd.concat(manova_S_reports, axis=0, ignore_index=True) if manova_S_reports else pd.DataFrame(columns=["Block","Background","N","Levels","Summary"])

    # ---- 11.6: Eksport ----------------------------------------------------------

    out_step11 = base_path.with_name(base_path.stem + "_ilr_spearman_manova.xlsx")
    with pd.ExcelWriter(out_step11, engine="xlsxwriter") as w:
        if rho_OO is not None:
            export_excel(rho_OO, writer=w, sheet_name="Spearman_ilrO_rho", label="Spearman_ilrO_rho")
            export_excel(p_OO,   writer=w, sheet_name="Spearman_ilrO_p",   label="Spearman_ilrO_p")
            long_OO = _spearman_matrix_to_long(rho_OO, p_OO, "OCAI", "ILR", "spearman", len(_ocai_ilr) if not _ocai_ilr.empty else None)
            if not long_OO.empty:
                export_excel(long_OO, writer=w, sheet_name="Sp_ilrO_LONG", label="Sp_ilrO_LONG")
                _log_sig_summary(long_OO, "Sp_ilrO_LONG", "STEG11")
        if rho_SS is not None:
            export_excel(rho_SS, writer=w, sheet_name="Spearman_ilrS_rho", label="Spearman_ilrS_rho")
            export_excel(p_SS,   writer=w, sheet_name="Spearman_ilrS_p",   label="Spearman_ilrS_p")
            long_SS = _spearman_matrix_to_long(rho_SS, p_SS, "Strategi", "ILR", "spearman", len(_strat_ilr) if not _strat_ilr.empty else None)
            if not long_SS.empty:
                export_excel(long_SS, writer=w, sheet_name="Sp_ilrS_LONG", label="Sp_ilrS_LONG")
                _log_sig_summary(long_SS, "Sp_ilrS_LONG", "STEG11")
        if rho_OS is not None:
            export_excel(rho_OS, writer=w, sheet_name="Spearman_cross_ilr_rho", label="Spearman_cross_ilr_rho")
            export_excel(p_OS,   writer=w, sheet_name="Spearman_cross_ilr_p",   label="Spearman_cross_ilr_p")
            long_OS = _spearman_matrix_to_long(rho_OS, p_OS, "OCAI_vs_Strategi", "ILR", "spearman", len(_ocai_ilr) if not _ocai_ilr.empty else None)
            if not long_OS.empty:
                export_excel(long_OS, writer=w, sheet_name="Sp_ilrX_LONG", label="Sp_ilrX_LONG")
                _log_sig_summary(long_OS, "Sp_ilrX_LONG", "STEG11")
        if not MANOVA_O.empty:
            manova_o = MANOVA_O.copy()
            if "method" not in manova_o.columns:
                manova_o.insert(1, "method", "MANOVA")
            if "space" not in manova_o.columns:
                manova_o.insert(2, "space", "ILR")
            export_excel(manova_o, writer=w, sheet_name="MANOVA_OCAI", label="MANOVA_OCAI")
        if not MANOVA_S.empty:
            manova_s = MANOVA_S.copy()
            if "method" not in manova_s.columns:
                manova_s.insert(1, "method", "MANOVA")
            if "space" not in manova_s.columns:
                manova_s.insert(2, "space", "ILR")
            export_excel(manova_s, writer=w, sheet_name="MANOVA_Strat", label="MANOVA_Strat")

        meta_rows_11 = []
        if rho_OO is not None:
            meta_rows_11.extend([
                {"outfile": out_step11.name, "sheet_name": "Spearman_ilrO_rho", "block": "OCAI", "space": "ILR", "method": "Spearman", "variables": "ilrO vs ilrO", "n_used": len(_ocai_ilr) if not _ocai_ilr.empty else ""},
                {"outfile": out_step11.name, "sheet_name": "Spearman_ilrO_p", "block": "OCAI", "space": "ILR", "method": "Spearman", "variables": "ilrO vs ilrO (p)", "n_used": len(_ocai_ilr) if not _ocai_ilr.empty else ""},
                {"outfile": out_step11.name, "sheet_name": "Sp_ilrO_LONG", "block": "OCAI", "space": "ILR", "method": "Spearman", "variables": "ilrO vs ilrO (long)", "n_used": len(_ocai_ilr) if not _ocai_ilr.empty else ""},
            ])
        if rho_SS is not None:
            meta_rows_11.extend([
                {"outfile": out_step11.name, "sheet_name": "Spearman_ilrS_rho", "block": "Strategi", "space": "ILR", "method": "Spearman", "variables": "ilrS vs ilrS", "n_used": len(_strat_ilr) if not _strat_ilr.empty else ""},
                {"outfile": out_step11.name, "sheet_name": "Spearman_ilrS_p", "block": "Strategi", "space": "ILR", "method": "Spearman", "variables": "ilrS vs ilrS (p)", "n_used": len(_strat_ilr) if not _strat_ilr.empty else ""},
                {"outfile": out_step11.name, "sheet_name": "Sp_ilrS_LONG", "block": "Strategi", "space": "ILR", "method": "Spearman", "variables": "ilrS vs ilrS (long)", "n_used": len(_strat_ilr) if not _strat_ilr.empty else ""},
            ])
        if rho_OS is not None:
            meta_rows_11.extend([
                {"outfile": out_step11.name, "sheet_name": "Spearman_cross_ilr_rho", "block": "OCAI_vs_Strategi", "space": "ILR", "method": "Spearman", "variables": "ilrO vs ilrS", "n_used": len(_ocai_ilr) if not _ocai_ilr.empty else ""},
                {"outfile": out_step11.name, "sheet_name": "Spearman_cross_ilr_p", "block": "OCAI_vs_Strategi", "space": "ILR", "method": "Spearman", "variables": "ilrO vs ilrS (p)", "n_used": len(_ocai_ilr) if not _ocai_ilr.empty else ""},
                {"outfile": out_step11.name, "sheet_name": "Sp_ilrX_LONG", "block": "OCAI_vs_Strategi", "space": "ILR", "method": "Spearman", "variables": "ilrO vs ilrS (long)", "n_used": len(_ocai_ilr) if not _ocai_ilr.empty else ""},
            ])
        if not MANOVA_O.empty:
            meta_rows_11.append({"outfile": out_step11.name, "sheet_name": "MANOVA_OCAI", "block": "OCAI", "space": "ILR", "method": "MANOVA", "variables": "ilrO by background", "n_used": ""})
        if not MANOVA_S.empty:
            meta_rows_11.append({"outfile": out_step11.name, "sheet_name": "MANOVA_Strat", "block": "Strategi", "space": "ILR", "method": "MANOVA", "variables": "ilrS by background", "n_used": ""})
        export_excel(pd.DataFrame(meta_rows_11), writer=w, sheet_name="RUN_META", label="RUN_META")

    print(f"[STEG 11] Lagret ILR-Spearman og MANOVA (sensitivitetsanalyse) til: {out_step11}")
    register_output(step="STEG 11", label="ilr_spearman_manova", path=out_step11, kind="xlsx")

    # Konsoll-oversikt av det som ble skrevet i STEG 11:
    if rho_OO is not None:
        _print_table("[STEG 11] Spearman ilrO – rho", rho_OO)
        _print_table("[STEG 11] Spearman ilrO – p",   p_OO)
    if rho_SS is not None:
        _print_table("[STEG 11] Spearman ilrS – rho", rho_SS)
        _print_table("[STEG 11] Spearman ilrS – p",   p_SS)
    if rho_OS is not None:
        _print_table("[STEG 11] Spearman (ilrO vs ilrS) – rho", rho_OS)
        _print_table("[STEG 11] Spearman (ilrO vs ilrS) – p",   p_OS)

    if not MANOVA_O.empty:
        _print_table("[STEG 11] MANOVA OCAI – sammendrag",
                     MANOVA_O[["Background","N","Levels"]])
    if not MANOVA_S.empty:
        _print_table("[STEG 11] MANOVA Strategi – sammendrag",
                     MANOVA_S[["Background","N","Levels"]])

    
    # Bruker modellene fra steg 10 til robuste t-tester og felles Wald-tester på ILR-koeffisienter og kontroller.
    # Input er models_S/models_O + designmatriser, output er <dataset>_ilr_beta_tests.xlsx og korte konsollutdrag.
    # ============================
    # STEG 12: Direkte tester på ILR-betakoeffisienter
    # ============================
    # Forutsetter fra STEG 10:
    #   - models_S, models_O (statsmodels OLS med cov_type="HC3")
    #   - X_all (design for Strategy ~ OCAI + controls)
    #   - X_rev (design for OCAI ~ Strategy + controls)
    #   - base_path
    # Denne delen lager:
    #   (a) "Coef_tests_t_HC3" – t-tester per koeffisient (HC3)
    #   (b) "Joint_Wald_HC3"   – felles Wald-tester for grupper (ILR-preds, Likert A–E, dummies, alle helninger)
    
    def _has_models_and_design():
        okS = isinstance(models_S, dict) and len(models_S) > 0 and not X_all.empty
        okO = isinstance(models_O, dict) and len(models_O) > 0 and not X_rev.empty
        return okS, okO
    
    def _order_by_resp_key(k):
        m = re.search(r'(\d+)$', str(k))
        return (int(m.group(1)) if m else 0, str(k))
    
    def _build_index_groups(X_cols, block='S'):
        """
        Definer grupper for felles Wald/F-tester:
          - ILR-prediktorer (fra motsatt blokk)
          - Likert-kontroller (A..E)
          - Hver bakgrunns-dummyfamilie
          - Alle helninger samlet (alle != const)
        """
        groups = []
        if block == 'S':
            ilr_pred = [i for i, c in enumerate(X_cols) if str(c).startswith("ilrO_")]
        else:
            ilr_pred = [i for i, c in enumerate(X_cols) if str(c).startswith("ilrS_")]
        if ilr_pred:
            groups.append(("ILR_predictors", ilr_pred))
    
        lik = [i for i, c in enumerate(X_cols) if c in ["A","B","C","D","E"]]
        if lik:
            groups.append(("Likert_A_E", lik))
    
        for stem in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]:
            idxs = [i for i, c in enumerate(X_cols) if str(c).startswith(stem + "_")]
            if idxs:
                groups.append((f"{stem}_dummies", idxs))
    
        all_idx = [i for i, c in enumerate(X_cols) if str(c) != "const"]
        if all_idx:
            groups.append(("All_slopes", all_idx))
        return groups

    _robust_fallback_counter = {"count": 0}

    def _get_robust_cov_with_fallback(m, prefer="HC3", fallback="HC1"):
        # k can differ from len(X_cols) if the fitted model dropped/added params (e.g., collinearity/const handling)
        try:
            k = int(np.asarray(getattr(m, "params", [])).shape[0])
        except Exception:
            k = 0
        if k == 0:
            try:
                k = int(np.asarray(m.model.exog).shape[1])
            except Exception:
                k = 0

        def _try_cov(cov_type):
            reasons = []
            try:
                m_rob = m.get_robustcov_results(cov_type=cov_type)
            except Exception:
                return None, None, None, None, None, [f"{cov_type}_get_robustcov_failed"]
            try:
                cov = np.asarray(m_rob.cov_params(), dtype=float)
            except Exception:
                cov = None
            try:
                se = np.asarray(m_rob.bse, dtype=float)
            except Exception:
                se = None

            if cov is None:
                reasons.append(f"{cov_type}_cov_none")
            if se is None:
                reasons.append(f"{cov_type}_se_none")

            if k == 0 and cov is not None and cov.ndim == 2 and cov.shape[0] == cov.shape[1]:
                k_eff = int(cov.shape[0])
            else:
                k_eff = k

            if cov is not None and (cov.ndim != 2 or (k_eff and cov.shape != (k_eff, k_eff))):
                reasons.append(f"{cov_type}_cov_bad_shape")
            if se is not None and (k_eff and se.shape != (k_eff,)):
                reasons.append(f"{cov_type}_se_bad_shape")
            if cov is not None and not np.isfinite(cov).all():
                reasons.append(f"{cov_type}_cov_nonfinite")
            if cov is not None and cov.ndim == 2:
                diag = np.diag(cov)
                if not np.isfinite(diag).all():
                    reasons.append(f"{cov_type}_cov_diag_nonfinite")
                if np.any(diag < 0):
                    reasons.append(f"{cov_type}_cov_diag_negative")
            if se is not None:
                if not np.isfinite(se).all():
                    reasons.append(f"{cov_type}_se_nonfinite")
                if np.all(np.isinf(se)):
                    reasons.append(f"{cov_type}_se_all_inf")
                if np.any(se == 0):
                    reasons.append(f"{cov_type}_se_zero")

            if reasons:
                return None, None, None, None, None, reasons

            try:
                tvals = np.asarray(m_rob.tvalues, dtype=float)
            except Exception:
                tvals = None
            try:
                pvals = np.asarray(m_rob.pvalues, dtype=float)
            except Exception:
                pvals = None
            if tvals is None or pvals is None:
                return None, None, None, None, None, [f"{cov_type}_t_p_missing"]

            return m_rob, cov, se, tvals, pvals, []

        m_rob, cov, se, tvals, pvals, reasons = _try_cov(prefer)
        if not reasons:
            return m_rob, cov, se, tvals, pvals, prefer, [], True

        m_rob_fb, cov_fb, se_fb, tvals_fb, pvals_fb, reasons_fb = _try_cov(fallback)
        if not reasons_fb:
            _robust_fallback_counter["count"] += 1
            return m_rob_fb, cov_fb, se_fb, tvals_fb, pvals_fb, fallback, reasons, True

        reasons.extend(reasons_fb)
        return None, None, None, None, None, "NONE", reasons, False
    
    def _coef_table_direct(mdict, X_cols, model_label):
        """Tidy-tabell med robuste (HC3) t-tester per beta (H0: beta=0) + supplerende modell-F."""
        rows = []
        p = len(X_cols)
        warned_global_test = False

        for resp, m in sorted(mdict.items(), key=lambda kv: _order_by_resp_key(kv[0])):
            # eksisterende per-koeff tall
            params  = np.asarray(m.params)
            m_rob, cov, se, tvals, pvals, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
                m, prefer="HC3", fallback="HC1"
            )
            if cov_stable and cov_type_used == "HC1" and RUN_MODE == "DEV":
                try:
                    nobs = int(m.nobs)
                    k = int(np.asarray(params).shape[0])
                    df_resid = int(m.df_resid)
                except Exception:
                    nobs = getattr(m, "nobs", np.nan)
                    k = len(params)
                    df_resid = getattr(m, "df_resid", np.nan)
                reason_txt = ";".join(cov_reasons) if cov_reasons else "HC3_invalid"
                print(f"[WARN] robustcov HC3 invalid -> using HC1 | {model_label} | {resp} | n={nobs} k={k} df={df_resid} | {reason_txt}")
            if not cov_stable:
                cov = None
                se = np.full_like(params, np.nan, dtype=float)
                tvals = np.full_like(params, np.nan, dtype=float)
                pvals = np.full_like(params, np.nan, dtype=float)
            R2      = float(m.rsquared)
            R2_adj  = float(m.rsquared_adj)

            # Stabilitetsvurdering for robust inferens
            unstable_reasons = []
            cov_arr = None
            if not cov_stable:
                unstable_reasons.extend(cov_reasons)
            else:
                try:
                    cov_arr = np.asarray(cov, dtype=float)
                except Exception:
                    cov_arr = None
                    unstable_reasons.append("cov=bad_array")
            if cov_arr is None:
                pass
            else:
                if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
                    unstable_reasons.append("cov=bad_shape")
                else:
                    if not np.isfinite(cov_arr).all():
                        unstable_reasons.append("cov=nonfinite")
                    diag = np.diag(cov_arr)
                    if not np.isfinite(diag).all():
                        unstable_reasons.append("cov_diag=nonfinite")
            if not np.isfinite(se).all():
                unstable_reasons.append("se=nonfinite")
            if np.any(se == 0):
                unstable_reasons.append("se=zero")
            if np.any(se < 1e-12):
                unstable_reasons.append("se=too_small")
            if np.any(se > 1e6):
                unstable_reasons.append("se=too_large")
            try:
                exog = np.asarray(m.model.exog)
                x_rank = np.linalg.matrix_rank(exog, tol=1e-10)
                if x_rank < p:
                    unstable_reasons.append("x_rank<p")
                svals = np.linalg.svd(exog, compute_uv=False)
                cond = float(svals[0] / svals[-1]) if svals.size and svals[-1] != 0 else np.inf
                if cond > 1e12:
                    unstable_reasons.append("cond>1e12")
            except Exception:
                unstable_reasons.append("x_rank/cond=error")
            model_stable = len(unstable_reasons) == 0
            unstable_reason = ";".join(unstable_reasons) if unstable_reasons else ""
            if not model_stable:
                print(f"[WARN] UNSTABLE inference: {model_label} resp={resp} reason={unstable_reason}")
                params = np.full(p, np.nan, dtype=float)
                se = np.full(p, np.nan, dtype=float)
                tvals = np.full(p, np.nan, dtype=float)
                pvals = np.full(p, np.nan, dtype=float)

            # Sett opp grupper for F/Wald
            block   = 'S' if model_label.startswith("Strategy") else 'O'
            groups  = _build_index_groups(X_cols, block=block)

            # All slopes (alle != const)
            all_idx = [i for i, c in enumerate(X_cols) if str(c) != "const"]
            if all_idx:
                R_all = np.zeros((len(all_idx), p))
                for r, j in enumerate(all_idx):
                    R_all[r, j] = 1.0
                if not model_stable:
                    F_all_fv = np.nan
                    F_all_p = np.nan
                    F_all_d1 = np.nan
                    F_all_d2 = np.nan
                    F_all_test_type = "UNSTABLE"
                    F_all_Wv = np.nan
                    F_all_Wp = np.nan
                else:
                    # F-test (robust, bruker samme kovarians) med chi2-fallback
                    try:
                        F_all = m.f_test(R_all, cov_p=cov)
                        F_all_fv = float(np.squeeze(F_all.fvalue))
                        F_all_p  = float(F_all.pvalue)
                        F_all_d1 = int(F_all.df_num)
                        F_all_d2 = int(F_all.df_denom)
                        F_all_test_type = "F"
                    except ValueError:
                        try:
                            F_all = m.wald_test(R_all, cov_p=cov, use_f=False, scalar=True)
                            F_all_fv = float(np.squeeze(F_all.statistic))
                            F_all_p  = float(F_all.pvalue)
                            F_all_d1 = int(F_all.df_num)
                            F_all_d2 = np.nan
                            F_all_test_type = "chi2"
                        except ValueError:
                            q = int(R_all.shape[0])
                            params = np.asarray(m.params).reshape(-1, 1)
                            rbeta = R_all @ params
                            cov_arr = np.asarray(cov, dtype=float)
                            cov_clean = np.nan_to_num(cov_arr, nan=0.0, posinf=0.0, neginf=0.0)
                            S = R_all @ cov_clean @ R_all.T
                            S = 0.5 * (S + S.T)
                            S = S + np.eye(q) * 1e-12
                            if not np.isfinite(S).all():
                                if not warned_global_test:
                                    print("[WARN] Global test skipped due to non-finite/unstable covariance")
                                    warned_global_test = True
                                F_all_fv = np.nan
                                F_all_p = np.nan
                                F_all_d1 = q
                                F_all_d2 = np.nan
                                F_all_test_type = "chi2_manual_failed"
                            else:
                                try:
                                    Sinv = np.linalg.pinv(S, rcond=1e-12, hermitian=True)
                                    stat = float(rbeta.T @ Sinv @ rbeta)
                                    if not np.isfinite(stat):
                                        raise ValueError("Non-finite chi2 statistic")
                                    F_all_fv = stat
                                    F_all_p = float(chi2.sf(stat, q))
                                    F_all_d1 = q
                                    F_all_d2 = np.nan
                                    F_all_test_type = "chi2_manual"
                                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                                    if not warned_global_test:
                                        print("[WARN] Global test skipped due to non-finite/unstable covariance")
                                        warned_global_test = True
                                    F_all_fv = np.nan
                                    F_all_p = np.nan
                                    F_all_d1 = q
                                    F_all_d2 = np.nan
                                    F_all_test_type = "chi2_manual_failed"
                    # Wald(F)-variant (beholdes som før) men må ikke krasje
                    try:
                        F_all_W  = m.wald_test(R_all, cov_p=cov, use_f=True, scalar=True)
                        F_all_Wv = float(np.squeeze(F_all_W.statistic))
                        F_all_Wp = float(F_all_W.pvalue)
                    except ValueError:
                        F_all_Wv = np.nan
                        F_all_Wp = np.nan
            else:
                F_all_fv = F_all_p = F_all_Wv = F_all_Wp = np.nan
                F_all_d1 = F_all_d2 = np.nan
                F_all_test_type = np.nan

            # Blokk-F for ILR-prediktorer og Likert-kontroller (dersom de finnes)
            idx_ilr = next((idxs for g, idxs in groups if g == "ILR_predictors"), [])
            idx_ctl = next((idxs for g, idxs in groups if g == "Likert_A_E"), [])
    
            def _f_from_idx(idxs):
                if not idxs:
                    return (np.nan, np.nan, np.nan, np.nan)
                R = np.zeros((len(idxs), p))
                for r, j in enumerate(idxs):
                    R[r, j] = 1.0
                try:
                    Ft = m.f_test(R, cov_p=cov)
                    return (float(np.squeeze(Ft.fvalue)),
                            int(Ft.df_num),
                            int(Ft.df_denom),
                            float(Ft.pvalue))
                except ValueError:
                    q = int(R.shape[0])
                    try:
                        Ft = m.wald_test(R, cov_p=cov, use_f=False, scalar=True)
                        return (float(np.squeeze(Ft.statistic)),
                                q,
                                np.nan,
                                float(Ft.pvalue))
                    except ValueError:
                        cov_arr = np.asarray(cov, dtype=float)
                        cov_clean = np.nan_to_num(cov_arr, nan=0.0, posinf=0.0, neginf=0.0)
                        S = R @ cov_clean @ R.T
                        S = 0.5 * (S + S.T)
                        S = S + np.eye(q) * 1e-12
                        if not np.isfinite(S).all():
                            return (np.nan, q, np.nan, np.nan)
                        params = np.asarray(m.params).reshape(-1, 1)
                        rbeta = R @ params
                        try:
                            Sinv = np.linalg.pinv(S, rcond=1e-12, hermitian=True)
                            stat = float((rbeta.T @ Sinv @ rbeta).item())
                            if not np.isfinite(stat):
                                raise ValueError("Non-finite chi2 statistic")
                            pval = float(chi2.sf(stat, q))
                            return (stat, q, np.nan, pval)
                        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                            return (np.nan, q, np.nan, np.nan)
    
            if not model_stable:
                F_ilr = d1_ilr = d2_ilr = p_ilr = np.nan
                F_ctl = d1_ctl = d2_ctl = p_ctl = np.nan
            else:
                F_ilr, d1_ilr, d2_ilr, p_ilr = _f_from_idx(idx_ilr)
                F_ctl, d1_ctl, d2_ctl, p_ctl = _f_from_idx(idx_ctl)

            # Skriv ut én rad per term (som før), men med supplerende modellkolonner
            for j, term in enumerate(X_cols):
                rows.append({
                    "model": model_label,
                    "response": resp,
                    "term": term,
                    "coef": params[j],
                    "se_HC3": se[j],
                    "t": tvals[j],
                    "p": pvals[j],
                    "R2": R2,
                    "R2_adj": R2_adj,
                    "model_stable": bool(model_stable),
                    "unstable_reason": unstable_reason,
                    "robust_type_used": cov_type_used,
                    "robust_cov_prefer": "HC3",
                    "robust_cov_used": cov_type_used,
                    "robust_cov_note": ("ok" if cov_stable else ";".join(cov_reasons) if cov_reasons else "HC3_invalid"),
                    "robust_cov_finite": bool(cov_stable),
                    # nye model-level tillegg
                    "F_all": F_all_fv,
                    "df1_all": F_all_d1,
                    "df2_all": F_all_d2,
                    "pF_all": F_all_p,
                    "F_all_test_type": F_all_test_type,
                    "F_all_Wald": F_all_Wv,
                    "pF_all_Wald": F_all_Wp,
                    "F_ILR": F_ilr,
                    "df1_ILR": d1_ilr,
                    "df2_ILR": d2_ilr,
                    "pF_ILR": p_ilr,
                    "F_ctrl": F_ctl,
                    "df1_ctrl": d1_ctl,
                    "df2_ctrl": d2_ctl,
                    "pF_ctrl": p_ctl,
                })
    
        out = pd.DataFrame(rows)
    
        def _stars(x):
            if pd.isna(x): return ""
            return "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
    
        out["sig"] = out["p"].apply(_stars)
        if "model_stable" in out.columns:
            out.loc[~out["model_stable"], "sig"] = ""
        return out
    
    def _wald_joint_table(mdict, X_cols, model_label):
        """
        Felles Wald-tester (robuste, bruker modellens m.cov_params() – HC3):
          H0: beta_j = 0 for ALLE j i gruppen  → chi^2(q), q=#restriksjoner.
        """
        rows = []
        p = len(X_cols)
        for resp, m in sorted(mdict.items(), key=lambda kv: _order_by_resp_key(kv[0])):
            block = 'S' if model_label.startswith("Strategy") else 'O'
            groups = _build_index_groups(X_cols, block=block)
            m_rob, cov, se, tvals, pvals, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
                m, prefer="HC3", fallback="HC1"
            )
            cov_note = "ok" if cov_stable else ";".join(cov_reasons) if cov_reasons else "HC3_invalid"
            for gname, idxs in groups:
                q = len(idxs)
                R = np.zeros((q, p))
                for r, j in enumerate(idxs):
                    R[r, j] = 1.0
                try:
                    if not cov_stable or cov is None:
                        raise ValueError("ustabil kovarians")
                    wt = m.wald_test(R, cov_p=cov, use_f=False, scalar=True)   # chi^2
                    stat = np.asarray(wt.statistic).item()                
                    pval = float(np.atleast_1d(wt.pvalue)[0])
                    concl = "Avvis H0 (p<0.05)" if pval < 0.05 else "Ikke avvis H0"
                except Exception as e:
                    stat, pval, concl = (np.nan, np.nan, f"Feil: {repr(e)}")
                rows.append({
                    "model": model_label,
                    "response": resp,
                    "group": gname,
                    "q": q,
                    "chi2": stat,
                    "p": pval,
                    "terms": ", ".join([str(X_cols[j]) for j in idxs]),
                    "conclusion": concl,
                    "robust_cov_prefer": "HC3",
                    "robust_cov_used": cov_type_used,
                    "robust_cov_note": cov_note,
                    "robust_cov_finite": bool(cov_stable),
                })
        return pd.DataFrame(rows)
    
    okS, okO = _has_models_and_design()
    if not (okS or okO):
        print("[STEG 12] Mangler modeller fra STEG 10 (models_S/models_O). Hopper over.")
    else:
        tables, joints = [], []
    
        if okS:
            Xcols_S = list(X_all.columns)
            coef_S  = _coef_table_direct(models_S, Xcols_S, "Strategy ~ OCAI + controls")
            joint_S = _wald_joint_table(models_S, Xcols_S, "Strategy ~ OCAI + controls")
            tables.append(coef_S); joints.append(joint_S)
    
        if okO:
            Xcols_O = list(X_rev.columns)
            coef_O  = _coef_table_direct(models_O, Xcols_O, "OCAI ~ Strategy + controls")
            joint_O = _wald_joint_table(models_O, Xcols_O, "OCAI ~ Strategy + controls")
            tables.append(coef_O); joints.append(joint_O)
    
        BETA_TESTS = pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()
        BETA_JOINT = pd.concat(joints, axis=0, ignore_index=True) if joints else pd.DataFrame()
    
        out_beta = base_path.with_name(base_path.stem + "_ilr_beta_tests.xlsx")
        with pd.ExcelWriter(out_beta, engine="xlsxwriter") as w:
            if not BETA_TESTS.empty:
                out_coef = BETA_TESTS.copy()
                for c in ["coef","se_HC3","t","p","R2","R2_adj",
                          "F_all","df1_all","df2_all","pF_all",
                          "F_all_Wald","pF_all_Wald",
                          "F_ILR","df1_ILR","df2_ILR","pF_ILR",
                          "F_ctrl","df1_ctrl","df2_ctrl","pF_ctrl"]:
                    if c in out_coef.columns:
                        out_coef[c] = pd.to_numeric(out_coef[c], errors="coerce").round(6)
                export_excel(out_coef, writer=w, sheet_name="Coef_tests_t_HC3", label="Coef_tests_t_HC3")
                _log_sig_summary(out_coef, "Coef_tests_t_HC3", "STEG12", cap=500)
            if not BETA_JOINT.empty:
                out_joint = BETA_JOINT.copy()
                for c in ["chi2","p"]:
                    if c in out_joint.columns:
                        out_joint[c] = pd.to_numeric(out_joint[c], errors="coerce").round(6)
                export_excel(out_joint, writer=w, sheet_name="Joint_Wald_HC3", label="Joint_Wald_HC3")
                _log_sig_summary(out_joint, "Joint_Wald_HC3", "STEG12", cap=500)
    
        print(f"[STEG 12] Skrev ILR-betattester til: {out_beta}")
        register_output(step="STEG 12", label="ilr_betatester", path=out_beta, kind="xlsx")
    
        # Kort konsoll-sammendrag (uendret)
        if okS and not BETA_TESTS.empty:
            print("\n[STEG 12] Sammendrag (Strategy ~ OCAI + controls):")
            _log_sig_summary(BETA_TESTS[BETA_TESTS["model"].str.startswith("Strategy")], "BETA_TESTS_Strategy", "STEG12")
        if okO and not BETA_TESTS.empty:
            print("\n[STEG 12] Sammendrag (OCAI ~ Strategy + controls):")
            _log_sig_summary(BETA_TESTS[BETA_TESTS["model"].str.startswith("OCAI")], "BETA_TESTS_OCAI", "STEG12")
    
    # Skriv Joint Wald (gruppetester) kompakt til konsoll (uendret)
    if not BETA_JOINT.empty:
        _log_sig_summary(
            BETA_JOINT[BETA_JOINT["model"].str.startswith("Strategy")],
            "Joint_Wald_Strategy",
            "STEG12",
        )
        _log_sig_summary(
            BETA_JOINT[BETA_JOINT["model"].str.startswith("OCAI")],
            "Joint_Wald_OCAI",
            "STEG12",
        )

    # DEV-only smoke check for suspicious inference patterns
    if RUN_MODE == "DEV":
        def _smoke_check_inference(mdict, X_cols, label):
            for resp, m in sorted(mdict.items(), key=lambda kv: _order_by_resp_key(kv[0])):
                # k can differ from len(X_cols) due to dropped/added params (e.g., collinearity or const handling)
                try:
                    k = int(np.asarray(getattr(m, "params", [])).shape[0])
                except Exception:
                    k = 0
                if k == 0:
                    try:
                        k = int(np.asarray(m.model.exog).shape[1])
                    except Exception:
                        k = 0
                m_rob, cov, se, tvals, pvals, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
                    m, prefer="HC3", fallback="HC1"
                )
                if not cov_stable:
                    se = np.full(k, np.nan, dtype=float)
                    pvals = np.full(k, np.nan, dtype=float)
                cov_finite = bool(cov_stable) and (cov is not None) and np.isfinite(np.asarray(cov, dtype=float)).all()

                se_nonfinite = np.any(~np.isfinite(se)) if k > 0 else False
                se_min = float(np.nanmin(se)) if se.size else np.nan
                se_max = float(np.nanmax(se)) if se.size else np.nan
                med_se = float(np.nanmedian(se)) if se.size else np.nan

                p_tiny_se = False
                share_p0 = np.nan
                if k > 0:
                    p_mask = np.isfinite(pvals)
                    if np.any(p_mask):
                        share_p0 = float(np.mean(pvals[p_mask] < 1e-15))
                    p_tiny_se = (np.isfinite(share_p0) and share_p0 > 0.8 and med_se < 1e-8)

                try:
                    exog = np.asarray(m.model.exog)
                    x_rank = int(np.linalg.matrix_rank(exog, tol=1e-10))
                    svals = np.linalg.svd(exog, compute_uv=False)
                    cond = float(svals[0] / svals[-1]) if svals.size and svals[-1] != 0 else np.inf
                except Exception:
                    x_rank = np.nan
                    cond = np.nan

                lev_min = lev_max = np.nan
                try:
                    infl = m.get_influence()
                    lev = np.asarray(getattr(infl, "hat_matrix_diag", np.nan))
                    lev_min = float(np.nanmin(lev)) if lev.size else np.nan
                    lev_max = float(np.nanmax(lev)) if lev.size else np.nan
                except Exception:
                    pass

                if se_nonfinite or p_tiny_se or (not cov_finite):
                    reason = []
                    if se_nonfinite:
                        reason.append("se_nonfinite")
                    if p_tiny_se:
                        reason.append("p_tiny_se")
                    if not cov_finite:
                        reason.append("cov_finite=False")
                    print(f"[DEV SMOKE] {label} resp={resp} reason={';'.join(reason)} "
                          f"se[min,max]=[{se_min:.2e},{se_max:.2e}] lev[min,max]=[{lev_min:.2e},{lev_max:.2e}] "
                          f"x_rank={x_rank} cond={cond:.2e}")

        if okS:
            _smoke_check_inference(models_S, Xcols_S, "Strategy ~ OCAI + controls")
        if okO:
            _smoke_check_inference(models_O, Xcols_O, "OCAI ~ Strategy + controls")
    
    # Refitter modellene med kun ILR-prediktorer (uten kontroller) for å teste blokk-sammenheng.
    # Input er designene fra steg 10 og modellobjektene; output er <dataset>_ilr_beta_tests_NOCTRL.xlsx og konsollrapporter.
    # ============================
    # STEG 12A: ILR-only (uten kontroller) — robust, bruker X_all/X_rev direkte
    # ============================
    # Forutsetter fra STEG 10:
    #   - models_S: dict med OLS for ilrS_k ~ [ilrO_*, A..E, dummies, const]
    #   - models_O: dict med OLS for ilrO_k ~ [ilrS_*, A..E, dummies, const]
    #   - X_all: design for Strategy ~ OCAI + controls (kolonner inkluderer ilrO_1..3)
    #   - X_rev: design for OCAI ~ Strategy + controls (kolonner inkluderer ilrS_1..3)
    #   - base_path
    # Lager:
    #   - eksempeldatasett_synthetic_mle_ilr_beta_tests_NOCTRL.xlsx
    #       * Coef_t_HC3_NOCTRL
    #       * Joint_Wald_HC3_NOCTRL
    
    def _has_models_12A():
        okS = isinstance(models_S, dict) and len(models_S) > 0 and not X_all.empty
        okO = isinstance(models_O, dict) and len(models_O) > 0 and not X_rev.empty
        return okS, okO
    
    def _order_by_resp_key(k):
        m = re.search(r'(\d+)$', str(k))
        return (int(m.group(1)) if m else 0, str(k))
    
    def _ilr_cols_from_design(design_df, prefix):
        # Strengt først: ^prefix\d+$ (ilrO_1 osv.); fallback: inneholder prefix
        strict = [c for c in design_df.columns if re.match(rf'^{re.escape(prefix)}\d+$', str(c))]
        if strict:
            return strict
        fallback = [c for c in design_df.columns if str(c).lower().startswith(prefix.lower())]
        return fallback
    
    def _refit_ilr_only_using_design(mdict, design_df, ilr_prefix, model_label):
        """
        Refitt hver responsmodell i mdict med KUN ILR-kolonner fra design_df (+ const).
        Antar samme rekkefølge/antall rader som i originalmodellene (syntetiske data → ok).
        """
        out = {}
        ilr_cols = _ilr_cols_from_design(design_df, ilr_prefix)
        if len(ilr_cols) == 0:
            print(f"[STEG 12A][ADVARSEL] Fant ingen kolonner som matcher '{ilr_prefix}*' i designet: {list(design_df.columns)}")
        Xsub_df = design_df[ilr_cols].copy() if ilr_cols else pd.DataFrame(index=design_df.index)
    
        # Legg på konstant (uansett)
        Xsub_df = sm.add_constant(Xsub_df, has_constant='add')
    
        for resp, m0 in sorted(mdict.items(), key=lambda kv: _order_by_resp_key(kv[0])):
            y = np.asarray(m0.model.endog)  # samme observasjoner/rekkefølge som i originalen
            X = np.asarray(Xsub_df)         # samme N antas
            term_cols = list(Xsub_df.columns)
            if X.shape[0] != y.shape[0]:
                # Sikkerhetsnett: hvis rader skulle avvike (bør ikke skje i den syntetiske pipelinen)
                print(f"[STEG 12A][ADVARSEL] Raduoverensstemmelse for '{resp}': X={X.shape[0]} vs y={y.shape[0]}. "
                      "Faller tilbake til intercept-only.")
                X = sm.add_constant(np.empty((y.shape[0], 0)), has_constant='add')
                term_cols = ["const"]
    
            res = sm.OLS(y, X).fit()
            # lagre navneliste for pene tabeller
            res._ilr_only_names = list(Xsub_df.columns)
            model_family, variant = _split_model_label(model_label)
            y_block, x_block = _infer_blocks_from_family(model_family)
            model_id = _make_model_id("STEG12A", model_family, variant, resp)
            stats = _fit_stats_dict(res)
            _log_modeldef(
                "STEG12A",
                model_id,
                model_family,
                variant,
                resp,
                y_block,
                x_block,
                "OLS",
                stats,
                term_cols,
            )
            _log_fitrow("STEG12A", model_id, model_family, variant, resp, "OLS", stats)
            _log_coefrows("STEG12A", model_id, model_family, variant, resp, term_cols, res, "OLS", stats)
            out[resp] = res
        return out
    
    def _coef_table(mdict, model_label):
        rows = []
        for resp, m in sorted(mdict.items(), key=lambda kv: _order_by_resp_key(kv[0])):
            p  = np.asarray(m.params)
            m_rob, cov, se, t, pv, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
                m, prefer="HC3", fallback="HC1"
            )
            if not cov_stable:
                se = np.full_like(p, np.nan, dtype=float)
                t = np.full_like(p, np.nan, dtype=float)
                pv = np.full_like(p, np.nan, dtype=float)
            cov_note = "ok" if cov_stable else ";".join(cov_reasons) if cov_reasons else "HC3_invalid"
            cols = getattr(m, "_ilr_only_names", m.model.exog_names)
            for j, term in enumerate(cols):
                rows.append({
                    "model": model_label,
                    "response": resp,
                    "term": term,
                    "coef": float(p[j]) if j < len(p) else np.nan,
                    "se_HC3": float(se[j]) if j < len(se) else np.nan,
                    "t": float(t[j]) if j < len(t) else np.nan,
                    "p": float(pv[j]) if j < len(pv) else np.nan,
                    "R2": m.rsquared,
                    "robust_cov_prefer": "HC3",
                    "robust_cov_used": cov_type_used,
                    "robust_cov_note": cov_note,
                    "robust_cov_finite": bool(cov_stable),
                })
        out = pd.DataFrame(rows)
        def _stars(x):
            if pd.isna(x): return ""
            return "***" if x < 0.001 else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
        out["sig"] = out["p"].apply(_stars)
        return out
    
    def _wald_ilr_block(mdict, model_label, ilr_prefix):
        rows = []
        for resp, m in sorted(mdict.items(), key=lambda kv: _order_by_resp_key(kv[0])):
            cols = list(getattr(m, "_ilr_only_names", m.model.exog_names))
            p = len(cols)
            m_rob, cov, se, tvals, pvals, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
                m, prefer="HC3", fallback="HC1"
            )
            cov_note = "ok" if cov_stable else ";".join(cov_reasons) if cov_reasons else "HC3_invalid"
            ilr_idx = [j for j, nm in enumerate(map(str, cols))
                       if nm != "const" and nm.lower().startswith(ilr_prefix.lower())]
            if len(ilr_idx) == 0:
                rows.append({
                    "model": model_label, "response": resp, "group": "ILR_block",
                    "q": 0, "chi2": np.nan, "p": np.nan, "terms": "", "conclusion": "Ingen ILR-termer",
                    "robust_cov_prefer": "HC3",
                    "robust_cov_used": cov_type_used,
                    "robust_cov_note": cov_note,
                    "robust_cov_finite": bool(cov_stable),
                })
                continue
            R = np.zeros((len(ilr_idx), p))
            for r, j in enumerate(ilr_idx): R[r, j] = 1.0
            try:
                if not cov_stable or cov is None:
                    raise ValueError("ustabil kovarians")
                wt = m.wald_test(R, cov_p=cov, use_f=False, scalar=True)  # chi^2
                stat = np.asarray(wt.statistic).item()            
                pval = float(np.atleast_1d(wt.pvalue)[0])
                concl = "Avvis H0 (p<0.05)" if pval < 0.05 else "Ikke avvis H0"
            except Exception as e:
                stat, pval, concl = (np.nan, np.nan, f"Feil: {repr(e)}")
            rows.append({
                "model": model_label, "response": resp, "group": "ILR_block",
                "q": len(ilr_idx), "chi2": stat, "p": pval,
                "terms": ", ".join([cols[j] for j in ilr_idx]), "conclusion": concl,
                "robust_cov_prefer": "HC3",
                "robust_cov_used": cov_type_used,
                "robust_cov_note": cov_note,
                "robust_cov_finite": bool(cov_stable),
            })
        return pd.DataFrame(rows)
    
    okS, okO = _has_models_12A()
    if not (okS or okO):
        print("[STEG 12A] Mangler modeller/design fra STEG 10. Hopper over.")
    else:
        fit_S_ilr = _refit_ilr_only_using_design(models_S, X_all, ilr_prefix="ilrO_", model_label="Strategy ~ OCAI (ILR only)") if okS else {}
        fit_O_ilr = _refit_ilr_only_using_design(models_O, X_rev, ilr_prefix="ilrS_", model_label="OCAI ~ Strategy (ILR only)") if okO else {}
    
        tables, joints = [], []
        if fit_S_ilr:
            coef_S = _coef_table(fit_S_ilr, "Strategy ~ OCAI (ILR only)")
            wald_S = _wald_ilr_block(fit_S_ilr, "Strategy ~ OCAI (ILR only)", ilr_prefix="ilrO_")
            tables.append(coef_S); joints.append(wald_S)
        if fit_O_ilr:
            coef_O = _coef_table(fit_O_ilr, "OCAI ~ Strategy (ILR only)")
            wald_O = _wald_ilr_block(fit_O_ilr, "OCAI ~ Strategy (ILR only)", ilr_prefix="ilrS_")
            tables.append(coef_O); joints.append(wald_O)
    
        COEF_NOCTRL  = pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()
        JOINT_NOCTRL = pd.concat(joints, axis=0, ignore_index=True) if joints else pd.DataFrame()
    
        out_beta_nc = base_path.with_name(base_path.stem + "_ilr_beta_tests_NOCTRL.xlsx")
        with pd.ExcelWriter(out_beta_nc, engine="xlsxwriter") as w:
            if not COEF_NOCTRL.empty:
                out_coef = COEF_NOCTRL.copy()
                for c in ["coef","se_HC3","t","p","R2"]:
                    if c in out_coef.columns:
                        out_coef[c] = pd.to_numeric(out_coef[c], errors="coerce").round(6)
                export_excel(out_coef, writer=w, sheet_name="Coef_t_HC3_NOCTRL", label="Coef_t_HC3_NOCTRL")
                _log_sig_summary(out_coef, "Coef_t_HC3_NOCTRL", "STEG12A", cap=500)
            if not JOINT_NOCTRL.empty:
                out_joint = JOINT_NOCTRL.copy()
                for c in ["chi2","p"]:
                    if c in out_joint.columns:
                        out_joint[c] = pd.to_numeric(out_joint[c], errors="coerce").round(6)
                export_excel(out_joint, writer=w, sheet_name="Joint_Wald_HC3_NOCTRL", label="Joint_Wald_HC3_NOCTRL")
                _log_sig_summary(out_joint, "Joint_Wald_HC3_NOCTRL", "STEG12A", cap=500)
    
        print(f"[STEG 12A] Skrev ILR-only betatester til: {out_beta_nc}")
        register_output(step="STEG 12A", label="ilr_only_betatester", path=out_beta_nc, kind="xlsx")
        if not COEF_NOCTRL.empty:
            print(f"[STEG 12A] Coef_t_HC3_NOCTRL: shape={COEF_NOCTRL.shape}, cols={list(COEF_NOCTRL.columns)}")
            _log_sig_summary(COEF_NOCTRL[COEF_NOCTRL["model"].str.startswith("Strategy")], "COEF_NOCTRL_Strategy", "STEG12A")
            _log_sig_summary(COEF_NOCTRL[COEF_NOCTRL["model"].str.startswith("OCAI")], "COEF_NOCTRL_OCAI", "STEG12A")

        if not JOINT_NOCTRL.empty:
            _log_sig_summary(JOINT_NOCTRL[JOINT_NOCTRL["model"].str.startswith("Strategy")], "JOINT_NOCTRL_Strategy", "STEG12A")
            _log_sig_summary(JOINT_NOCTRL[JOINT_NOCTRL["model"].str.startswith("OCAI")], "JOINT_NOCTRL_OCAI", "STEG12A")
    
    
    
    
    # Transformerer ILR-betakoeffisienter til CLR-rommet og beregner tilhørende SE/t/Wald for tolkning.
    # Input er ILR-modellene, Ψ-matriser og profiler; output er CLR-tabeller i Excel og utskrift.
    # ============================
    # STEG 12B: CLR-ekvivalente betakoeffisienter og tester (fra ILR)
    # ============================
    # Forutsetter fra STEG 10/12:
    #   - models_S, models_O : dict[str -> statsmodels.OLSResults] med cov_type="HC3"
    #   - X_all, X_rev       : designmatriser brukt for modellene
    #   - _psi_pivot_isolate_first(first_idx=0)    (Ψ, shape 4x3)
    #   - ocai_cols, strat_cols, base_path
    # Bruker lineær transformasjon:
    #   z = clr @ Ψ              (allerede brukt i STEG 10)
    #   => clr = z @ Ψᵀ
    #   β_clr = Ψ β_ilr
    #   Σ_clr = Ψ Σ_ilr Ψᵀ,  SE_clr = sqrt(diag(Σ_clr))
    
    def _extract_ilr_block(m, X_cols, ilr_prefix):
        """Returner (beta_ilr (3,), Sigma_ilr (3x3), ilr_idx(list), cov_type_used, cov_note, cov_stable) for ILR-prediktorblokken."""
        cols = list(map(str, X_cols))
        ilr_idx = [j for j, nm in enumerate(cols) if nm != "const" and nm.lower().startswith(ilr_prefix.lower())]
        if len(ilr_idx) != 3:
            return None, None, ilr_idx, "NONE", "ilr_idx_len!=3", False
        beta = np.asarray(m.params)[ilr_idx]
        m_rob, cov, se, tvals, pvals, cov_type_used, cov_reasons, cov_stable = _get_robust_cov_with_fallback(
            m, prefer="HC3", fallback="HC1"
        )
        cov_note = "ok" if cov_stable else ";".join(cov_reasons) if cov_reasons else "HC3_invalid"
        if not cov_stable or cov is None:
            return beta, None, ilr_idx, cov_type_used, cov_note, False
        Sigma = np.asarray(cov)[np.ix_(ilr_idx, ilr_idx)]
        return beta, Sigma, ilr_idx, cov_type_used, cov_note, True
    
    def _clr_from_ilr(beta_ilr, Sigma_ilr, psi_4x3):
        """β_clr (4,), Σ_clr (4x4), SE_clr (4,), t_clr (4,), p_clr (4,)"""
        B = np.asarray(beta_ilr).reshape(3,1)          # (3,1)
        S = np.asarray(Sigma_ilr).reshape(3,3)         # (3,3)
        S_clean = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
        PSI = np.asarray(psi_4x3)                      # (4,3)

        beta_clr = (PSI @ B).ravel()                   # (4,)
        if not np.isfinite(S).all():
            Sigma_clr = np.full((4,4), np.nan, dtype=float)
            se_clr = np.full(4, np.nan, dtype=float)
            t_clr = np.full(4, np.nan, dtype=float)
            p_clr = np.full(4, np.nan, dtype=float)
            return beta_clr, Sigma_clr, se_clr, t_clr, p_clr
        Sigma_clr = PSI @ S_clean @ PSI.T              # (4,4)
        Sigma_clr = 0.5 * (Sigma_clr + Sigma_clr.T)
        Sigma_clr = Sigma_clr + np.eye(Sigma_clr.shape[0]) * 1e-12
        se_clr = np.sqrt(np.clip(np.diag(Sigma_clr), 0.0, np.inf))
        # t-stat ~ N(0,1) robust (samme som i ILR-blokken), p tosidig:
        t_clr = np.divide(beta_clr, se_clr, out=np.full_like(beta_clr, np.nan, dtype=float), where=se_clr>0)
        p_clr = 2.0 * (1.0 - norm.cdf(np.abs(t_clr)))
        return beta_clr, Sigma_clr, se_clr, t_clr, p_clr
    
    def _wald_group_clr(beta_clr, Sigma_clr):
        """
        Felles Wald for alle CLR-helninger (H0: alle β_clr=0).
        Returnerer (stat, q, p, conclusion, note) der note forklarer ev. degradering/pseudoinvers.
        """
        b = np.asarray(beta_clr, dtype=float).reshape(4, 1)
        S = np.asarray(Sigma_clr, dtype=float).reshape(4, 4)
        q = 3  # 4 parts => D-1
        # grunnleggende kontroller
        if (not np.isfinite(b).all()) or (not np.isfinite(S).all()):
            return np.nan, q, np.nan, "NOT COMPUTABLE", "non-finite beta/cov"
        try:
            rank_S = np.linalg.matrix_rank(S, tol=1e-10)
            svals = np.linalg.svd(S, compute_uv=False)
            smin = float(svals[-1]) if svals.size else np.nan
        except np.linalg.LinAlgError as e:
            return np.nan, q, np.nan, "NOT COMPUTABLE", f"svd/rank error {repr(e)}"
        if rank_S < q:
            return np.nan, q, np.nan, "NOT COMPUTABLE", f"rank({rank_S})<q, smin={smin:.2e}"
        try:
            Sinv = np.linalg.pinv(S, rcond=1e-12)
            stat = np.asarray(b.T @ Sinv @ b, dtype=float).squeeze().item()
            pval = float(_chi2.sf(stat, q))
            concl = "Avvis H0 (p<0.05)" if pval < 0.05 else "Ikke avvis H0"
            # For CLR med D=4 er rank=q forventet; pinv er da normal, ikke en degradering
            note = "pinv expected (CLR rank=D-1)" if (rank_S == q) else ("pinv (ill-cond)" if (smin < 1e-8) else "ok")
            return stat, q, pval, concl, note
        except Exception as e:
            return np.nan, q, np.nan, "NOT COMPUTABLE", f"exception {repr(e)}"
    
    def _clr_tables_for_models(mdict, X_cols, ilr_prefix, psi, block_label, prof_names):
        """
        Bygger to DataFrames:
          - COEF_CLR: per response (ilrY_k) × profil (4): coef_clr, se_clr, t, p
          - JOINT_CLR: per response: felles Wald (CLR-blokken, q=3)
        """
        rows_coef, rows_joint = [], []
        for resp_key, m in sorted(mdict.items(), key=lambda kv: int(str(kv[0]).split('_')[-1]) if str(kv[0]).split('_')[-1].isdigit() else 0):
            beta_ilr, Sigma_ilr, ilr_idx, cov_type_used, cov_note, cov_stable = _extract_ilr_block(m, X_cols, ilr_prefix)
            if beta_ilr is None:
                rows_joint.append({
                    "model": block_label, "response": resp_key, "q": len(ilr_idx),
                    "chi2": np.nan, "p": np.nan, "conclusion": "Mangler ILR-blokk (forventet 3)",
                    "clr_stable": False, "clr_note": "missing ILR block",
                    "clr_cov_finite": False, "clr_rank": np.nan,
                    "clr_min_sval": np.nan, "clr_max_sval": np.nan, "clr_cond": np.nan,
                    "clr_transform_shape": "", "clr_block_terms": "",
                    "clr_ilr_block_rank": np.nan, "clr_ilr_block_p": np.nan, "clr_nobs": np.nan,
                    "robust_cov_prefer": "HC3", "robust_cov_used": cov_type_used,
                    "robust_cov_note": cov_note, "robust_cov_finite": bool(cov_stable),
                })
                continue

            # diagnostikk før CLR/Wald
            try:
                exog = np.asarray(m.model.exog)
                x_rank = np.linalg.matrix_rank(exog, tol=1e-10)
                svals = np.linalg.svd(exog, compute_uv=False)
                cond = float(svals[0]/svals[-1]) if svals.size and svals[-1] != 0 else np.inf
            except Exception:
                exog = None
                x_rank = np.nan
                cond = np.nan
            cov_ilr = np.asarray(Sigma_ilr) if Sigma_ilr is not None else None
            cov_finite = bool(cov_stable) and (cov_ilr is not None) and np.isfinite(cov_ilr).all()
            try:
                clr_nobs = float(m.nobs)
            except Exception:
                clr_nobs = np.nan
            clr_ilr_block_p = len(ilr_idx)
            clr_ilr_block_rank = np.nan
            if exog is not None and ilr_idx:
                try:
                    X_ilr = exog[:, ilr_idx]
                    svals_ilr = np.linalg.svd(X_ilr, compute_uv=False)
                    smax_ilr = float(svals_ilr[0]) if svals_ilr.size else np.nan
                    tol_ilr = (smax_ilr * 1e-12) if np.isfinite(smax_ilr) else 0.0
                    clr_ilr_block_rank = int(np.sum(svals_ilr > tol_ilr)) if svals_ilr.size else np.nan
                except Exception:
                    clr_ilr_block_rank = np.nan
            print(f"[STEG 12B diag] {block_label} resp={resp_key} nobs={clr_nobs} df_model={m.df_model} df_resid={m.df_resid} "
                  f"x_rank={x_rank} cond={cond} cov_finite={cov_finite} ilr_p={clr_ilr_block_p} ilr_rank={clr_ilr_block_rank}")

            beta_clr, Sigma_clr, se_clr, t_clr, p_clr = _clr_from_ilr(beta_ilr, Sigma_ilr if Sigma_ilr is not None else np.full((3,3), np.nan), psi)
            clr_reasons = []
            if not cov_finite:
                clr_reasons.append("cov_finite=False")
            if not np.isfinite(Sigma_clr).all():
                clr_reasons.append("Sigma_clr=nonfinite")
            try:
                rank_Sclr = np.linalg.matrix_rank(Sigma_clr, tol=1e-10)
                svals_Sclr = np.linalg.svd(Sigma_clr, compute_uv=False)
                smin_Sclr = float(svals_Sclr[-1]) if svals_Sclr.size else np.nan
                smax_Sclr = float(svals_Sclr[0]) if svals_Sclr.size else np.nan
            except Exception:
                rank_Sclr = np.nan
                smin_Sclr = np.nan
                smax_Sclr = np.nan
                clr_reasons.append("svd/rank error")
            clr_min_sval = smin_Sclr
            clr_max_sval = smax_Sclr
            clr_cond = (smax_Sclr / smin_Sclr) if (np.isfinite(smax_Sclr) and np.isfinite(smin_Sclr) and smin_Sclr > 0) else np.nan
            clr_transform_shape = f"PSI:{np.asarray(psi).shape};cov_ilr:{np.asarray(cov_ilr).shape if cov_ilr is not None else None};cov_clr:{np.asarray(Sigma_clr).shape}"
            clr_block_terms = ", ".join([str(X_cols[j]) for j in ilr_idx]) if ilr_idx else ""
            if np.isfinite(rank_Sclr) and rank_Sclr < 3:
                clr_reasons.append(f"rank<{3}")
            clr_stable = len(clr_reasons) == 0
            clr_note = ";".join(clr_reasons) if clr_reasons else "ok"

            if not clr_stable:
                print("[WARN] CLR diagnostics skipped due to non-finite/degenerate covariance")
                se_clr = np.full(4, np.nan, dtype=float)
                t_clr = np.full(4, np.nan, dtype=float)
                p_clr = np.full(4, np.nan, dtype=float)
                stat = np.nan
                q = 3
                pval = np.nan
                concl = f"NOT COMPUTABLE ({clr_note})"
                note = clr_note
            else:
                stat, q, pval, concl, note = _wald_group_clr(beta_clr, Sigma_clr)
                if np.isnan(stat) or np.isnan(pval):
                    concl = f"NOT COMPUTABLE ({note}; rank_Sclr={rank_Sclr}, smin={smin_Sclr:.2e})"
                print(f"[STEG 12B diag] {block_label} resp={resp_key} Sigma_clr_rank={rank_Sclr} smin={smin_Sclr:.2e} "
                      f"chi2={stat} p={pval} note={note}")
    
            for j, prof in enumerate(prof_names):
                coef_val = beta_clr[j] if np.isfinite(beta_clr[j]) else np.nan
                rows_coef.append({
                    "model": block_label,
                    "response": resp_key,
                    "clr_term": f"CLR({prof})",
                    "coef_clr": coef_val,
                    "se_clr": se_clr[j],
                    "t_clr": t_clr[j],
                    "p_clr": p_clr[j],
                    "clr_stable": bool(clr_stable),
                    "clr_note": clr_note,
                    "clr_cov_finite": bool(cov_finite),
                    "clr_rank": rank_Sclr,
                    "clr_min_sval": clr_min_sval,
                    "clr_max_sval": clr_max_sval,
                    "clr_cond": clr_cond,
                    "clr_transform_shape": clr_transform_shape,
                    "clr_block_terms": clr_block_terms,
                    "clr_ilr_block_rank": clr_ilr_block_rank,
                    "clr_ilr_block_p": clr_ilr_block_p,
                    "clr_nobs": clr_nobs,
                    "robust_cov_prefer": "HC3",
                    "robust_cov_used": cov_type_used,
                    "robust_cov_note": cov_note,
                    "robust_cov_finite": bool(cov_stable),
                })
            rows_joint.append({
                "model": block_label, "response": resp_key,
                "q": q, "chi2": stat, "p": pval,
                "conclusion": concl,
                "note": note,
                "clr_stable": bool(clr_stable),
                "clr_note": clr_note,
                "clr_cov_finite": bool(cov_finite),
                "clr_rank": rank_Sclr,
                "clr_min_sval": clr_min_sval,
                "clr_max_sval": clr_max_sval,
                "clr_cond": clr_cond,
                "clr_transform_shape": clr_transform_shape,
                "clr_block_terms": clr_block_terms,
                "clr_ilr_block_rank": clr_ilr_block_rank,
                "clr_ilr_block_p": clr_ilr_block_p,
                "clr_nobs": clr_nobs,
                "robust_cov_prefer": "HC3",
                "robust_cov_used": cov_type_used,
                "robust_cov_note": cov_note,
                "robust_cov_finite": bool(cov_stable),
            })
        return pd.DataFrame(rows_coef), pd.DataFrame(rows_joint)
    
    # ---- Kjører transformasjonen for begge retninger ------------------------------
    
    okS_12 = isinstance(models_S, dict) and len(models_S) > 0 and not X_all.empty
    okO_12 = isinstance(models_O, dict) and len(models_O) > 0 and not X_rev.empty
    if not (okS_12 or okO_12):
        print("[STEG 12B] Mangler modeller fra STEG 10/12. Hopper over.")
    else:
        # Ψ for hver blokk (samme pivot som i regresjonene dine)
        PSI_O_4x3 = _psi_pivot_isolate_first(first_idx=0)  # prediktor-Ψ når OCAI er på X-siden
        PSI_S_4x3 = _psi_pivot_isolate_first(first_idx=0)  # prediktor-Ψ når Strategi er på X-siden
    
        tables, joints = [], []
    
        if okS_12:
            # Strategy ~ OCAI + controls → ilrO_* er prediktorer → CLR for OCAI-profiler
            Xcols_S = list(X_all.columns)
            coef_S_CLR, wald_S_CLR = _clr_tables_for_models(
                models_S, Xcols_S, ilr_prefix="ilrO_", psi=PSI_O_4x3,
                block_label="Strategy ~ OCAI (CLR-ekv.)",
                prof_names=ocai_cols
            )
            tables.append(coef_S_CLR); joints.append(wald_S_CLR)
    
        if okO_12:
            # OCAI ~ Strategy + controls → ilrS_* er prediktorer → CLR for Strategi-profiler
            Xcols_O = list(X_rev.columns)
            coef_O_CLR, wald_O_CLR = _clr_tables_for_models(
                models_O, Xcols_O, ilr_prefix="ilrS_", psi=PSI_S_4x3,
                block_label="OCAI ~ Strategy (CLR-ekv.)",
                prof_names=strat_cols
            )
            tables.append(coef_O_CLR); joints.append(wald_O_CLR)
    
        COEF_CLR = pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()
        JOINT_CLR = pd.concat(joints, axis=0, ignore_index=True) if joints else pd.DataFrame()
    
        out_clr = base_path.with_name(base_path.stem + "_clr_from_ilr_beta_tests.xlsx")
        with pd.ExcelWriter(out_clr, engine="xlsxwriter") as w:
            if not COEF_CLR.empty:
                out_coef = COEF_CLR.copy()
                for c in ["coef_clr","se_clr","t_clr","p_clr"]:
                    out_coef[c] = pd.to_numeric(out_coef[c], errors="coerce").round(6)
                export_excel(out_coef, writer=w, sheet_name="Coef_CLR_from_ILR", label="Coef_CLR_from_ILR")
                _log_sig_summary(
                    out_coef.rename(columns={"p_clr": "p", "coef_clr": "coef", "clr_term": "term"}),
                    "Coef_CLR_from_ILR",
                    "STEG12B",
                    cap=500,
                )
            if not JOINT_CLR.empty:
                out_joint = JOINT_CLR.copy()
                for c in ["chi2","p"]:
                    out_joint[c] = pd.to_numeric(out_joint[c], errors="coerce").round(6)
                export_excel(out_joint, writer=w, sheet_name="Joint_Wald_CLR", label="Joint_Wald_CLR")
                _log_sig_summary(out_joint, "Joint_Wald_CLR", "STEG12B", cap=500)
    
        print(f"[STEG 12B] Skrev CLR-ekvivalente koeffisienter/SE/t/Wald til: {out_clr}")
        register_output(step="STEG 12B", label="clr_equiv", path=out_clr, kind="xlsx")
        if not COEF_CLR.empty:
            print("[STEG 12B] Coef_CLR_from_ILR: shape=", COEF_CLR.shape, "cols=", list(COEF_CLR.columns))
            _log_sig_summary(COEF_CLR.rename(columns={"p_clr": "p", "coef_clr": "coef", "clr_term": "term"}), "COEF_CLR", "STEG12B")
        if not JOINT_CLR.empty:
            print("[STEG 12B] Joint_Wald_CLR – sammendrag:")
            _log_sig_summary(JOINT_CLR, "JOINT_CLR", "STEG12B")
    
    
    
    
    # Validerer CLR-beregningene fra forrige steg med algebraiske sjekker og evt. bootstrap/delta-sammenligning.
    # Input er modeller/Ψ/ILR→CLR-objekter, output er et sanity-check-Excel og konsollvarsler.
    # ============================
    # STEG 12C: Sanity checks for CLR-β og SE (ILR → CLR)
    # ============================
    # Forutsetter fra STEG 10/12/12B:
    #   - _psi_pivot_isolate_first (Ψ, 4x3)
    #   - models_S, models_O    : dict[str -> statsmodels.OLSResults]
    #   - X_all, X_rev          : designmatriser brukt i regresjonene
    #   - Z_S, Z_O              : ILR-responser (Strategy / OCAI)
    #   - ocai_cols, strat_cols : profilnavn (4)
    #   - base_path             : grunnsti for ut-filer
    #
    # Hensikt:
    #   (A) Algebraiske invariants:
    #       - ΨᵀΨ ≈ I
    #       - sum(β_clr) ≈ 0
    #       - Σ_clr har rad-/kolonnesummer ≈ 0
    #       - round-trip: β_ilr ≈ Ψᵀ β_clr, Σ_ilr ≈ Ψᵀ Σ_clr Ψ
    #   (B) Pairs bootstrap (B=200) for CLR-β:
    #       - sammenligner SE_clr (delta-metode) vs. SE_clr (bootstrap)
    #
    
    okS_12C = isinstance(models_S, dict) and len(models_S) > 0 and not Z_S.empty and not X_all.empty
    okO_12C = isinstance(models_O, dict) and len(models_O) > 0 and not Z_O.empty and not X_rev.empty
    
    if not (okS_12C or okO_12C):
        print("[STEG 12C] Mangler nødvendige objekter (models_*, Z_*, X_*). Hopper over konsistenskontroller.")
    else:
        # ---------- 12C.1: Algebraiske invariants for Ψ og CLR-mapping ----------
        PSI = _psi_pivot_isolate_first(first_idx=0)
        I_approx = PSI.T @ PSI
        I_err = np.linalg.norm(I_approx - np.eye(3), ord="fro")
        
        # --- Eksplicitte utskrifter for identitetssjekk ---
        print("[STEG 12C] Identity check for Ψ: Psi^T Psi (skal være ≈ I3):")
        print(np.round(I_approx, 6))
        print(f"[STEG 12C] Frobenius-norm av (Psi^T Psi - I3): {I_err:.6e}")
        
        # --- Enkel OK/ADVARSEL basert på toleranse ---
        tol = 1e-10
        if I_err < tol:
            print(f"[STEG 12C] OK: Psi^T Psi er ortonormalt innenfor toleranse {tol}.")
        else:
            print(f"[STEG 12C] ADVARSEL: Psi^T Psi avviker fra identitet med Frobenius-norm {I_err:.3e} (> {tol}).")
    
        inv_rows = []
        def _collect_invariants(mdict, X_cols, psi, ilr_prefix, block_label):
            for resp_key, m in mdict.items():
                beta_ilr, Sigma_ilr, ilr_idx, cov_type_used, cov_note, cov_stable = _extract_ilr_block(m, X_cols, ilr_prefix)
                if beta_ilr is None or Sigma_ilr is None:
                    inv_rows.append({
                        "block": block_label,
                        "response": resp_key,
                        "sum_beta_clr": np.nan,
                        "max_row_sum_Sigma_clr": np.nan,
                        "max_col_sum_Sigma_clr": np.nan,
                        "max_abs_roundtrip_beta_ilr": np.nan,
                        "max_abs_roundtrip_Sigma_ilr": np.nan,
                    })
                    continue
                beta_clr, Sigma_clr, se_clr, _, _ = _clr_from_ilr(beta_ilr, Sigma_ilr, psi)
    
                # Sum-to-zero i CLR
                sum_beta = float(np.sum(beta_clr))
                row_sums = np.sum(Sigma_clr, axis=1)
                col_sums = np.sum(Sigma_clr, axis=0)
    
                # Round-trip tilbake til ILR
                beta_ilr_rt = psi.T @ beta_clr.reshape(-1, 1)
                Sigma_ilr_rt = psi.T @ Sigma_clr @ psi
    
                max_diff_beta = float(np.max(np.abs(beta_ilr_rt.ravel() - np.asarray(beta_ilr).ravel())))
                max_diff_Sigma = float(np.max(np.abs(Sigma_ilr_rt - np.asarray(Sigma_ilr))))
    
                inv_rows.append({
                    "block": block_label,
                    "response": resp_key,
                    "sum_beta_clr": sum_beta,
                    "max_row_sum_Sigma_clr": float(np.max(np.abs(row_sums))),
                    "max_col_sum_Sigma_clr": float(np.max(np.abs(col_sums))),
                    "max_abs_roundtrip_beta_ilr": max_diff_beta,
                    "max_abs_roundtrip_Sigma_ilr": max_diff_Sigma,
                })
    
        if okS_12C:
            _collect_invariants(
                models_S,
                X_cols=list(X_all.columns),
                psi=PSI,
                ilr_prefix="ilrO_",
                block_label="Strategy ~ OCAI (CLR fra OCAI-ILR)"
            )
        if okO_12C:
            _collect_invariants(
                models_O,
                X_cols=list(X_rev.columns),
                psi=PSI,
                ilr_prefix="ilrS_",
                block_label="OCAI ~ Strategy (CLR fra Strat-ILR)"
            )
    
        INV_DF = pd.DataFrame(inv_rows)
        INV_DF.insert(0, "Psi_Frobenius_norm_of_(PsiT_Psi-I3)", I_err)
    
        # ---------- 12C.2: Pairs bootstrap for CLR-SE (B=200) ----------
        B_BOOT = 200
        rng = np.random.default_rng(12345)
        boot_rows = []
    
        def _bootstrap_clr_SE(Z_df, X_df, mdict, X_cols, psi, ilr_prefix, prof_names, block_label):
            X = X_df.to_numpy()
            n = X.shape[0]
            for resp_key, m in mdict.items():
                # Finn original ILR-blokk og CLR-SE (delta-metode)
                beta_ilr, Sigma_ilr, ilr_idx, cov_type_used, cov_note, cov_stable = _extract_ilr_block(m, X_cols, ilr_prefix)
                if beta_ilr is None or Sigma_ilr is None:
                    continue
                beta_clr, Sigma_clr, se_clr_delta, _, _ = _clr_from_ilr(beta_ilr, Sigma_ilr, psi)
    
                # Responsvektor
                if resp_key not in Z_df.columns:
                    continue
                y = Z_df[resp_key].to_numpy()
    
                # Bootstrap for CLR-β
                beta_clr_boot = np.empty((B_BOOT, 4))
                beta_clr_boot[:] = np.nan
    
                for b in range(B_BOOT):
                    idx = rng.integers(0, n, size=n)
                    y_b = y[idx]
                    X_b = X[idx, :]
    
                    try:
                        m_b = sm.OLS(y_b, X_b).fit(cov_type="HC3")
                        model_family, variant_base = _split_model_label(block_label)
                        variant = f"{variant_base};boot={b}"
                        y_block, x_block = _infer_blocks_from_family(model_family)
                        model_id = _make_model_id("STEG12C", model_family, variant, resp_key)
                        stats = _fit_stats_dict(m_b)
                        _log_modeldef(
                            "STEG12C",
                            model_id,
                            model_family,
                            variant,
                            resp_key,
                            y_block,
                            x_block,
                            "HC3",
                            stats,
                            X_cols,
                        )
                        _log_fitrow("STEG12C", model_id, model_family, variant, resp_key, "HC3", stats)
                        _log_coefrows("STEG12C", model_id, model_family, variant, resp_key, X_cols, m_b, "HC3", stats)
                        beta_ilr_b, Sigma_ilr_b, ilr_idx_b, cov_type_used_b, cov_note_b, cov_stable_b = _extract_ilr_block(m_b, X_cols, ilr_prefix)
                        if beta_ilr_b is None or Sigma_ilr_b is None:
                            continue
                        beta_clr_b, _, _, _, _ = _clr_from_ilr(beta_ilr_b, Sigma_ilr_b, psi)
                        beta_clr_boot[b, :] = beta_clr_b
                    except Exception:
                        # Hopper over eventuelle numeriske problemer i enkelt-bootstrap
                        continue
    
                se_clr_boot = np.nanstd(beta_clr_boot, axis=0, ddof=1)
    
                for j in range(4):
                    boot_rows.append({
                        "block": block_label,
                        "response": resp_key,
                        "clr_prof": prof_names[j] if j < len(prof_names) else f"CLR_{j+1}",
                        "se_clr_delta": float(se_clr_delta[j]),
                        "se_clr_boot": float(se_clr_boot[j]),
                        "ratio_boot_over_delta": float(se_clr_boot[j] / se_clr_delta[j]) if se_clr_delta[j] > 0 else np.nan,
                    })
    
        if okS_12C:
            _bootstrap_clr_SE(
                Z_df=Z_S,
                X_df=X_all,
                mdict=models_S,
                X_cols=list(X_all.columns),
                psi=PSI,
                ilr_prefix="ilrO_",
                prof_names=ocai_cols,
                block_label="Strategy ~ OCAI (CLR fra OCAI-ILR)"
            )
    
        if okO_12C:
            _bootstrap_clr_SE(
                Z_df=Z_O,
                X_df=X_rev,
                mdict=models_O,
                X_cols=list(X_rev.columns),
                psi=PSI,
                ilr_prefix="ilrS_",
                prof_names=strat_cols,
                block_label="OCAI ~ Strategy (CLR fra Strat-ILR)"
            )
    
        BOOT_DF = pd.DataFrame(boot_rows)
    
        # ---------- 12C.3: Eksport + konsoll-logg ----------
        out_checks = base_path.with_name(base_path.stem + "_clr_sanity_checks.xlsx")
        with pd.ExcelWriter(out_checks, engine="xlsxwriter") as w:
            if not INV_DF.empty:
                export_excel(INV_DF, writer=w, sheet_name="algebraic_invariants", label="algebraic_invariants")
            if not BOOT_DF.empty:
                export_excel(BOOT_DF, writer=w, sheet_name="bootstrap_SE_CLR", label="bootstrap_SE_CLR")
    
        print(f"[STEG 12C] Skrev CLR-konsistenskontroller til: {out_checks}")
        register_output(step="STEG 12C", label="clr_sanity", path=out_checks, kind="xlsx")
        if not INV_DF.empty:
            print("[STEG 12C] Algebraiske invariants – shape=", INV_DF.shape, "cols=", list(INV_DF.columns))
            _log_df_meta("[STEG 12C] INV_DF (meta)", INV_DF)
        if not BOOT_DF.empty:
            print("[STEG 12C] Bootstrap vs delta-SE (CLR) – shape=", BOOT_DF.shape, "cols=", list(BOOT_DF.columns))
            _log_df_meta("[STEG 12C] BOOT_DF (meta)", BOOT_DF)
    
    # Kjører sekvensielle modeller der kontroller og bakgrunn fjernes/blokkeres for å teste robusthet i ILR-sammenhenger.
    # Input er modeller/design fra steg 10 og helperne fra 12; output er blokkreduksjons-tabeller på Excel/konsoll.
    # ============================
    # STEG 12D: Sekvensielle blokkreduksjonsmodeller
    # ============================
    # Hensikt:
    #   - Refitt ILR-modellene med:
    #       (1) ILR + Likert (uten bakgrunnsdummier)
    #       (2) ILR + bakgrunnsdummier (uten Likert A–E)
    # - Bruk av samme logikk som STEG 12 (t-tester + felles Wald) uten å finne på BG_* navn.
    #   - Skriver til: <eksempeldatasett>_ilr_beta_tests_BLOCKREDUCED.xlsx
    #
    # Forutsetter:
    #   - models_S, models_O : dict[str -> statsmodels.OLSResults] (HC3) fra STEG 10
    #   - X_all, X_rev       : designmatriser for hhv. Strategy- og OCAI-modellene
    #   - _coef_table_direct, _wald_joint_table, _build_index_groups fra STEG 12
    #   - base_path
    
    def _has_models_12D():
        okS = isinstance(models_S, dict) and len(models_S) > 0 and not X_all.empty
        okO = isinstance(models_O, dict) and len(models_O) > 0 and not X_rev.empty
        return okS, okO
    
    def _split_design_blocks_12D(X_cols, ilr_prefix):
        """
        Identifiser ILR-, Likert- og bakgrunnskolonner i en designmatrise.
        - ilr_prefix: 'ilrO_' for Strategy-retningen, 'ilrS_' for OCAI-retningen.
        """
        cols_str = list(map(str, X_cols))
    
        ilr_idx = [j for j, c in enumerate(cols_str) if c.startswith(ilr_prefix)]
        lik_idx = [j for j, c in enumerate(cols_str) if c in ["A","B","C","D","E"]]
    
        bg_stems = ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]
        bg_idx = [
            j for j, c in enumerate(cols_str)
            if any(c.startswith(stem + "_") for stem in bg_stems)
        ]
    
        const_idx = [j for j, c in enumerate(cols_str) if c == "const"]
    
        return {
            "ilr": ilr_idx,
            "likert": lik_idx,
            "bg": bg_idx,
            "const": const_idx,
            "cols": cols_str,
        }
    
    def _refit_reduced_models_12D(mdict, X_df, ilr_prefix, base_label):
        """
        Refitt modeller i mdict med:
          - 'noBG'  : ILR + Likert (+ const)
          - 'noLik' : ILR + BG (+ const)
    
        Returnerer:
          - dict(spec -> (mdict_spec, X_cols_spec, label_spec))
        """
        blocks = _split_design_blocks_12D(X_df.columns, ilr_prefix=ilr_prefix)
        cols_str = blocks["cols"]
        ilr_idx  = set(blocks["ilr"])
        lik_idx  = set(blocks["likert"])
        bg_idx   = set(blocks["bg"])
        const_idx = set(blocks["const"])
    
        specs = {}
    
        def _cols_from_idx(idx_set, include_const=True):
            idx = set(idx_set)
            if include_const:
                idx |= const_idx
            idx = sorted(idx)
            return [cols_str[j] for j in idx]
    
        # (1) ILR + Likert (uten BG)
        if ilr_idx or lik_idx:
            cols_noBG = _cols_from_idx(ilr_idx | lik_idx, include_const=True)
            X_noBG_df = X_df[cols_noBG].copy()
            m_noBG = {}
            for resp_key, m0 in sorted(mdict.items(), key=lambda kv: int(str(kv[0]).split('_')[-1]) if str(kv[0]).split('_')[-1].isdigit() else 0):
                y = np.asarray(m0.model.endog)
                X = X_noBG_df.to_numpy()
                if X.shape[0] != y.shape[0]:
                    # Dette skal ikke skje i pipeline, men sikkerhetsnett:
                    print(f"[STEG 12D][ADVARSEL] Raduoverensstemmelse (noBG) for {base_label} / '{resp_key}': "
                          f"X={X.shape[0]} vs y={y.shape[0]}. Hopper over denne responsen.")
                    continue
                res = sm.OLS(y, X).fit()
                model_family, variant = _split_model_label(base_label + " (ILR + Likert, uten BG)")
                y_block, x_block = _infer_blocks_from_family(model_family)
                model_id = _make_model_id("STEG12D", model_family, variant, resp_key)
                stats = _fit_stats_dict(res)
                _log_modeldef(
                    "STEG12D",
                    model_id,
                    model_family,
                    variant,
                    resp_key,
                    y_block,
                    x_block,
                    "OLS",
                    stats,
                    cols_noBG,
                )
                _log_fitrow("STEG12D", model_id, model_family, variant, resp_key, "OLS", stats)
                _log_coefrows("STEG12D", model_id, model_family, variant, resp_key, cols_noBG, res, "OLS", stats)
                m_noBG[resp_key] = res
            specs["noBG"] = (m_noBG, cols_noBG, base_label + " (ILR + Likert, uten BG)")
    
        # (2) ILR + BG (uten Likert)
        if ilr_idx or bg_idx:
            cols_noLik = _cols_from_idx(ilr_idx | bg_idx, include_const=True)
            X_noLik_df = X_df[cols_noLik].copy()
            m_noLik = {}
            for resp_key, m0 in sorted(mdict.items(), key=lambda kv: int(str(kv[0]).split('_')[-1]) if str(kv[0]).split('_')[-1].isdigit() else 0):
                y = np.asarray(m0.model.endog)
                X = X_noLik_df.to_numpy()
                if X.shape[0] != y.shape[0]:
                    print(f"[STEG 12D][ADVARSEL] Raduoverensstemmelse (noLik) for {base_label} / '{resp_key}': "
                          f"X={X.shape[0]} vs y={y.shape[0]}. Hopper over denne responsen.")
                    continue
                res = sm.OLS(y, X).fit()
                model_family, variant = _split_model_label(base_label + " (ILR + BG, uten Likert)")
                y_block, x_block = _infer_blocks_from_family(model_family)
                model_id = _make_model_id("STEG12D", model_family, variant, resp_key)
                stats = _fit_stats_dict(res)
                _log_modeldef(
                    "STEG12D",
                    model_id,
                    model_family,
                    variant,
                    resp_key,
                    y_block,
                    x_block,
                    "OLS",
                    stats,
                    cols_noLik,
                )
                _log_fitrow("STEG12D", model_id, model_family, variant, resp_key, "OLS", stats)
                _log_coefrows("STEG12D", model_id, model_family, variant, resp_key, cols_noLik, res, "OLS", stats)
                m_noLik[resp_key] = res
            specs["noLik"] = (m_noLik, cols_noLik, base_label + " (ILR + BG, uten Likert)")
    
        return specs
    
    okS_12D, okO_12D = _has_models_12D()
    if not (okS_12D or okO_12D):
        print("[STEG 12D] Mangler modeller/design fra STEG 10. Hopper over blokkreduksjon.")
    else:
        coef_tables_12D = []
        joint_tables_12D = []
    
        # Strategy ~ OCAI-retningen
        if okS_12D:
            specs_S = _refit_reduced_models_12D(
                mdict=models_S,
                X_df=X_all,
                ilr_prefix="ilrO_",
                base_label="Strategy ~ OCAI"
            )
            for spec_name, (mdict_spec, X_cols_spec, label_spec) in specs_S.items():
                if not mdict_spec:
                    continue
                coef_spec = _coef_table_direct(mdict_spec, X_cols_spec, label_spec)
                joint_spec = _wald_joint_table(mdict_spec, X_cols_spec, label_spec)
                coef_spec.insert(0, "spec", spec_name)
                joint_spec.insert(0, "spec", spec_name)
                coef_tables_12D.append(coef_spec)
                joint_tables_12D.append(joint_spec)
    
        # OCAI ~ Strategy-retningen
        if okO_12D:
            specs_O = _refit_reduced_models_12D(
                mdict=models_O,
                X_df=X_rev,
                ilr_prefix="ilrS_",
                base_label="OCAI ~ Strategy"
            )
            for spec_name, (mdict_spec, X_cols_spec, label_spec) in specs_O.items():
                if not mdict_spec:
                    continue
                coef_spec = _coef_table_direct(mdict_spec, X_cols_spec, label_spec)
                joint_spec = _wald_joint_table(mdict_spec, X_cols_spec, label_spec)
                coef_spec.insert(0, "spec", spec_name)
                joint_spec.insert(0, "spec", spec_name)
                coef_tables_12D.append(coef_spec)
                joint_tables_12D.append(joint_spec)
    
        BETA_BLOCKRED = pd.concat(coef_tables_12D, axis=0, ignore_index=True) if coef_tables_12D else pd.DataFrame()
        WALD_BLOCKRED = pd.concat(joint_tables_12D, axis=0, ignore_index=True) if joint_tables_12D else pd.DataFrame()
    
        out_beta_block = base_path.with_name(base_path.stem + "_ilr_beta_tests_BLOCKREDUCED.xlsx")
        with pd.ExcelWriter(out_beta_block, engine="xlsxwriter") as w:
            if not BETA_BLOCKRED.empty:
                out_coef = BETA_BLOCKRED.copy()
                for c in ["coef","se_HC3","t","p","R2","R2_adj",
                          "F_all","df1_all","df2_all","pF_all",
                          "F_all_Wald","pF_all_Wald",
                          "F_ILR","df1_ILR","df2_ILR","pF_ILR",
                          "F_ctrl","df1_ctrl","df2_ctrl","pF_ctrl"]:
                    if c in out_coef.columns:
                        out_coef[c] = pd.to_numeric(out_coef[c], errors="coerce").round(6)
                export_excel(out_coef, writer=w, sheet_name="Coef_t_HC3_BLOCKRED", label="Coef_t_HC3_BLOCKRED")
                _log_sig_summary(out_coef, "Coef_t_HC3_BLOCKRED", "STEG12D", cap=500)
            if not WALD_BLOCKRED.empty:
                out_joint = WALD_BLOCKRED.copy()
                for c in ["chi2","p"]:
                    if c in out_joint.columns:
                        out_joint[c] = pd.to_numeric(out_joint[c], errors="coerce").round(6)
                export_excel(out_joint, writer=w, sheet_name="Joint_Wald_BLOCKRED", label="Joint_Wald_BLOCKRED")
                _log_sig_summary(out_joint, "Joint_Wald_BLOCKRED", "STEG12D", cap=500)
    
        print(f"[STEG 12D] Skrev blokkreduksjons-betatester til: {out_beta_block}")
        register_output(step="STEG 12D", label="block_reduction_betatester", path=out_beta_block, kind="xlsx")
    
        _log_df_meta("[STEG 12D] Coef_t_HC3_BLOCKRED (meta)", BETA_BLOCKRED)
        _log_df_meta("[STEG 12D] Joint_Wald_BLOCKRED (meta)", WALD_BLOCKRED)
        if not BETA_BLOCKRED.empty:
            _log_sig_summary(BETA_BLOCKRED, "BETA_BLOCKRED", "STEG12D")
        if not WALD_BLOCKRED.empty:
            _log_sig_summary(WALD_BLOCKRED, "WALD_BLOCKRED", "STEG12D")
        
    
        # Leser seksjonsark (dominerende/strategiske/suksess) per blokk, beregner ILR-baserte reliabilitetsmål og lager tetrafigurer.
        # Input er xlsx_path + DIR_RES/p_tables for plott; output er reliabilitets-Excel og flere PNG-er.
        # ============================
        # STEG 13: ILR-reliabilitet (seksjoner) + tetra-visualiseringer
        # ============================
        # Forutsetter: xlsx_path, base_path, ocai_cols, strat_cols, targets, find_sheet_name
        # (fra tidligere steg), samt (for visualisering) DIR_RES, _empirical_matrix_for_sheet,
        # _alpha_for_sheet fra STEG 8. Alle funksjonsnavn i denne blokken er suffikset med "13"
        # for å unngå kollisjon med tidligere steg.
        
        # ---------- 13.0: Seksjons-labels ----------
        _SEC_LABELS_13 = ["dominerende", "strategiske", "suksess"]
        
        # ---------- 13.1: Pivot-ILR basis og første balanse (profil k vs resten) ----------
        def _psi_pivot_isolate_first13(first_idx=0):
            """4x3 pivot-ILR der første koord isolerer valgt profil mot resten."""
            PSI = np.array([
                [ np.sqrt(3/4),         0.0,                 0.0               ],  # p0 vs (p1,p2,p3)
                [-1/np.sqrt(12),  np.sqrt(2/3),              0.0               ],  # p1 vs (p2,p3)
                [-1/np.sqrt(12), -1/np.sqrt(6),        1/np.sqrt(2)           ],  # p2 vs p3
                [-1/np.sqrt(12), -1/np.sqrt(6),       -1/np.sqrt(2)           ],
            ])
            if first_idx == 0:
                return PSI
            order = np.r_[first_idx, np.delete(np.arange(4), first_idx)]
            return PSI[order, :]
        
        def _first_balance_vector13(df_section, profile_cols, k_idx, eps=1e-12):
            """
            Returner DF med unik ID og z1 for 'profil k vs resten' i en seksjon.
            Viktig: aggregerer per ID med Aitchison-mean (clr-mean) før ILR, slik at flere rader per ID ikke dobbelttelles.
            """
            have = [c for c in profile_cols if c in df_section.columns]
            if len(have) != 4 or "ID" not in df_section.columns:
                return None
        
            tmp = df_section[["ID"] + have].copy()
            tmp["ID"] = tmp["ID"].astype(str)
            for c in have:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp = tmp.dropna(subset=have)
            if tmp.empty:
                return None
        
            X = tmp[have].to_numpy(float, copy=True)
            if np.nanmax(X) > 1.0:
                X = X / 100.0
        
            s = np.nansum(X, axis=1)
            keep = np.isfinite(s) & (np.abs(s - 1.0) <= 1e-6)
            if not np.any(keep):
                return None
        
            tmp = tmp.loc[keep].reset_index(drop=True)
            X = tmp[have].to_numpy(float, copy=True)
            X[X <= 0] = eps
            X = X / X.sum(axis=1, keepdims=True)
        
            # Aitchison-mean per ID via clr-mean
            logX = np.log(X)
            clr  = logX - logX.mean(axis=1, keepdims=True)
            clr_df = pd.DataFrame(clr, columns=have)
            clr_df.insert(0, "ID", tmp["ID"].to_numpy())
        
            clr_mean = clr_df.groupby("ID")[have].mean()  # (n_id, 4)
            clrM = clr_mean.to_numpy(float)
        
            PSI = _psi_pivot_isolate_first13(first_idx=k_idx)
            Z = clrM @ PSI  # (n_id, 3)
        
            out = clr_mean.reset_index()[["ID"]].copy()
            out["z1"] = Z[:, 0]
            return out
        
        # ---------- 13.2: Finn seksjonsark ----------
        def _find_section_sheets13(block_prefix):
            xls = pd.ExcelFile(xlsx_path)
            found = {}
            for lab in _SEC_LABELS_13:
                want = f"{block_prefix} - {lab}"
                sh = find_sheet_name(want, xls.sheet_names)
                if sh is not None:
                    found[lab] = sh
            return found
        
        # ---------- 13.3: ILR-reliabilitet per blokk ----------
        def _ilr_reliability_per_block13(block_name, profile_cols):
            """
            Bygger 'profil k vs resten' i tre seksjoner per respondent (via ID),
            og beregner r̄, Cronbach α (fra r̄) og split-half (Spearman–Brown).
            """
            prefix = "OCAI" if block_name == "OCAI" else "Strategi"
            sec_sh = _find_section_sheets13(prefix)
            if len(sec_sh) < 3:
                print(f"[STEG 13] Mangler seksjonsark for {block_name}: fant {list(sec_sh.values())}")
                return pd.DataFrame(), {}
        
            frames_raw = {}
            for lab in _SEC_LABELS_13:
                sh = sec_sh.get(lab, None)
                if sh is None:
                    print(f"[STEG 13] Seksjon mangler: {lab}")
                    return pd.DataFrame(), {}
                df = pd.read_excel(xlsx_path, sheet_name=sh)
                if "ID" not in df.columns:
                    print(f"[STEG 13] Skipper {sh}: mangler ID.")
                    return pd.DataFrame(), {}
                frames_raw[lab] = df.copy()
        
            results = []
            item_stats = {}
        
            for k_idx, prof in enumerate(profile_cols):
                per_sec = []
                for lab in _SEC_LABELS_13:
                    df = frames_raw[lab]
                    sec_df = _first_balance_vector13(df, profile_cols, k_idx)
                    if sec_df is None or sec_df.empty:
                        per_sec = []
                        break
                    sec_df = sec_df.rename(columns={"z1": f"b_{lab}"})
                    per_sec.append(sec_df)
        
                if len(per_sec) != 3:
                    continue
        
                M = per_sec[0].merge(per_sec[1], on="ID").merge(per_sec[2], on="ID")
                if M.shape[0] < 5:
                    a_corr = rbar = sb = np.nan
                    istd = pd.DataFrame({"item": _SEC_LABELS_13, "mean": [np.nan]*3, "sd": [np.nan]*3})
                    R = pd.DataFrame(np.full((3,3), np.nan), index=_SEC_LABELS_13, columns=_SEC_LABELS_13)
                else:
                    X = M[[f"b_{lab}" for lab in _SEC_LABELS_13]].to_numpy(float)
                    R = pd.DataFrame(X).corr()
                    r_vals = R.values[np.triu_indices(3, 1)]
                    rbar = float(np.nanmean(r_vals))
                    a_corr = (3*rbar)/(1+2*rbar) if np.isfinite(rbar) else np.nan
                    vals = []
                    for i in range(3):
                        xi = X[:, i]
                        xmean = X[:, [j for j in range(3) if j != i]].mean(axis=1)
                        r = np.corrcoef(xi, xmean)[0, 1]
                        if np.isfinite(r):
                            vals.append((2*r)/(1+r))
                    sb = float(np.mean(vals)) if vals else np.nan
                    istd = pd.DataFrame({
                        "item": _SEC_LABELS_13,
                        "mean": X.mean(axis=0),
                        "sd":   X.std(axis=0, ddof=1),
                    })
        
                results.append({
                    "Block": block_name,
                    "Profile": prof,
                    "N_used": int(M.shape[0]),
                    "avg_inter_item_r_ilr": rbar,
                    "alpha_from_r_ilr": a_corr,
                    "split_half_SB_ilr": sb
                })
                item_stats[prof] = {"item_stats": istd, "R": R}
        
            return pd.DataFrame(results), item_stats
        
        # Kjører for OCAI og Strategi, og skriver til fil
        rel_O_ilr13, item_O_ilr13 = _ilr_reliability_per_block13("OCAI", ocai_cols)
        rel_S_ilr13, item_S_ilr13 = _ilr_reliability_per_block13("Strategi", strat_cols)
        
        out_rel_ilr13 = xlsx_path.with_name(xlsx_path.stem + "_reliability_across_sections_ILR.xlsx")
        with pd.ExcelWriter(out_rel_ilr13, engine="xlsxwriter") as w:
            if not rel_O_ilr13.empty:
                export_excel(rel_O_ilr13, writer=w, sheet_name="OCAI_reliability_ILR", label="OCAI_reliability_ILR")
                for prof, packs in item_O_ilr13.items():
                    export_excel(packs["item_stats"], writer=w, sheet_name=f"OCAI_{prof[:26]}_ILR_items", label=f"OCAI_{prof[:26]}_ILR_items")
                    export_excel(packs["R"], writer=w, sheet_name=f"OCAI_{prof[:28]}_ILR_R", label=f"OCAI_{prof[:28]}_ILR_R")
            if not rel_S_ilr13.empty:
                export_excel(rel_S_ilr13, writer=w, sheet_name="Strategi_reliability_ILR", label="Strategi_reliability_ILR")
                for prof, packs in item_S_ilr13.items():
                    export_excel(packs["item_stats"], writer=w, sheet_name=f"STRAT_{prof[:25]}_ILR_items", label=f"STRAT_{prof[:25]}_ILR_items")
                    export_excel(packs["R"], writer=w, sheet_name=f"STRAT_{prof[:27]}_ILR_R", label=f"STRAT_{prof[:27]}_ILR_R")
        
        print(f"[STEG 13] Skrev ILR-basert tverr-seksjons reliabilitet til: {out_rel_ilr13}")
        register_output(step="STEG 13", label="ilr_reliability_cross_section", path=out_rel_ilr13, kind="xlsx")
        
        # ---------- 13.4: Tetra-geometri (egen navne-space for STEG 13) ----------
        def _tetra_vertices13():
            v1 = np.array([ 1.0,  0.0,  0.0])
            v2 = np.array([-1/3,  2*np.sqrt(2)/3,  0.0])
            v3 = np.array([-1/3, -np.sqrt(2)/3,  np.sqrt(6)/3])
            v4 = np.array([-1/3, -np.sqrt(2)/3, -np.sqrt(6)/3])
            return np.vstack([v1, v2, v3, v4])
        
        _V13 = _tetra_vertices13()
        _centroid13 = _V13.mean(axis=0)
        
        def _draw_tetra_wire13(ax):
            edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
            for i,j in edges:
                ax.plot([_V13[i,0],_V13[j,0]], [_V13[i,1],_V13[j,1]], [_V13[i,2],_V13[j,2]], linewidth=1, alpha=0.45)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        
        def _plot_tetra13(ax, labels):
            _draw_tetra_wire13(ax)
            for i, lab in enumerate(labels):
                ax.scatter(_V13[i,0], _V13[i,1], _V13[i,2], s=110, marker='o', alpha=0.9)
                ax.text(_V13[i,0], _V13[i,1], _V13[i,2], "  "+lab, fontsize=9, ha='left', va='bottom')
        
        def _bary_to_xyz13(P_4col: np.ndarray) -> np.ndarray:
            return P_4col @ _V13
        
        def _safe_name13(s: str) -> str:
            return ''.join(ch if ch.isalnum() or ch in ('-_') else '_' for ch in str(s))
        
        # ---------- 13.5: Pivot-ILR forward/inverse for fargelegging ----------
        def _pivot_ilr_forward13(P):
            P = np.asarray(P, float)
            P = np.clip(P, 1e-15, 1.0)
            p1,p2,p3,p4 = P[:,0], P[:,1], P[:,2], P[:,3]
            z1 = np.sqrt(3/4.0) * (np.log(p1) - (np.log(p2)+np.log(p3)+np.log(p4))/3.0)
            z2 = np.sqrt(2/3.0) * (np.log(p2) - (np.log(p3)+np.log(p4))/2.0)
            z3 = np.sqrt(1/2.0) * (np.log(p3) - np.log(p4))
            return np.column_stack([z1,z2,z3])
        
        def _pivot_ilr_inverse13(Z):
            Z = np.atleast_2d(np.asarray(Z, float))
            a = np.exp(Z[:,0] * (2/np.sqrt(3)))
            b = np.exp(Z[:,1] * np.sqrt(3/2))
            c = np.exp(Z[:,2] * np.sqrt(2))
            p1_prop = a * (b**(1/3)) * (c**(1/2))
            p2_prop = b * (c**(1/2))
            p3_prop = c
            p4_prop = np.ones_like(c)
            P = np.stack([p1_prop, p2_prop, p3_prop, p4_prop], axis=1)
            P = np.clip(P, 1e-15, np.inf)
            return P / P.sum(axis=1, keepdims=True)
        
        def _aitchison_center_rows13(P):
            eps = 1e-15
            X = np.clip(np.asarray(P, float), eps, None)
            X = X / X.sum(axis=1, keepdims=True)
            logX = np.log(X)
            m = np.mean(logX, axis=0)
            p0 = np.exp(m)
            p0 = p0 / p0.sum()
            return p0
        
        def _add_pivot_ilr_axes(ax, ilr_forward, ilr_inverse, comp_to_xyz, anchor_comp, vertices, plot_label):
            is_bary = anchor_comp is None
            if anchor_comp is None:
                anchor_comp = np.ones(4, dtype=float) / 4.0
            c0 = np.asarray(anchor_comp, float)
            c0 = np.clip(c0, 1e-15, np.inf)
            c0 = c0 / c0.sum()
            z0 = ilr_forward(c0.reshape(1, -1))[0]
            eps = 0.3
            edge_len = float(np.linalg.norm(vertices[0] - vertices[1]))
            arrow_len = 0.3 * edge_len
            delta = 0.03 * edge_len
            jitter = [
                np.array([delta, 0.0, 0.0]),
                np.array([0.0, delta, 0.0]),
                np.array([0.0, 0.0, delta]),
            ]
            xyz0 = comp_to_xyz(c0.reshape(1, -1))[0]
            for i in range(3):
                zp = z0.copy(); zp[i] += eps
                zm = z0.copy(); zm[i] -= eps
                cp = ilr_inverse(zp.reshape(1, -1))[0]
                cm = ilr_inverse(zm.reshape(1, -1))[0]
                v = comp_to_xyz(cp.reshape(1, -1))[0] - comp_to_xyz(cm.reshape(1, -1))[0]
                vn = np.linalg.norm(v)
                if not np.isfinite(vn) or vn == 0:
                    continue
                v = v / vn * arrow_len
                ax.quiver(xyz0[0], xyz0[1], xyz0[2], v[0], v[1], v[2],
                          arrow_length_ratio=0.08, linewidth=2, color='black')
                tip = xyz0 + v + jitter[i]
                ax.text(tip[0], tip[1], tip[2], f"+z{i+1}", fontsize=10, color='black')
            anchor_label = "bary" if is_bary else "aitchison"
            print(f"[ILR AXES] plot={plot_label} anchor={anchor_label} arrow_len={arrow_len:.4f} eps={eps}")
        
        # ---------- 13.6: Metode-demo (syntetisk) + Resultat-plott (reelle data) ----------
        def _methods_tetra_demo13(out_dir, labels, N=800, alpha=(40,20,20,20), color_by="z1"):
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(42)
            A = np.array(alpha, dtype=float)
            P = rng.dirichlet(A, size=N)
            Z = _pivot_ilr_forward13(P)
            xyz = _bary_to_xyz13(P)
            fig = plt.figure(figsize=(6.2, 6))
            ax = fig.add_subplot(111, projection='3d')
            _plot_tetra13(ax, labels)
            _add_pivot_ilr_axes(
                ax=ax,
                ilr_forward=_pivot_ilr_forward13,
                ilr_inverse=_pivot_ilr_inverse13,
                comp_to_xyz=_bary_to_xyz13,
                anchor_comp=None,
                vertices=_V13,
                plot_label="DEMO13",
            )
            idx = {"z1":0, "z2":1, "z3":2}.get(color_by, 0)
            sc = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=10, alpha=0.85, marker='o', c=Z[:,idx])
            cb = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02); cb.set_label(f"{color_by} (pivot-ILR)")
            ax.set_title(f"Metode-demo: Dirichlet(alpha={list(alpha)}) — pivot-ILR")
            out = out_dir / "metode_synthetic_tetra_pivotILR.png"
            plt.tight_layout(); plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
            print(f"[STEG 13A] Lagret: {out}")
            register_output(step="STEG 13A", label="result_fig", path=out, kind="png")
        
        def _overlay_observed_and_dirichlet13(ax, block, sheet, point_size=10, sim_mult=2, seed=123):
            if DIR_RES.empty:
                return False, None
            X = _empirical_matrix_for_sheet(sheet, block)
            a, _S = _alpha_for_sheet(DIR_RES, block, sheet)
            if X is None or a is None:
                return False, None
            xyz_emp = _bary_to_xyz13(X)
            n_obs = int(xyz_emp.shape[0])
            ax.scatter(xyz_emp[:,0], xyz_emp[:,1], xyz_emp[:,2], s=point_size, alpha=0.45, marker='o', label=f"Observed (n={n_obs})")
            _source = _DIRICHLET_SOURCE_MAP.get((block, sheet), "UNKNOWN")
            _add_pivot_ilr_axes(
                ax=ax,
                ilr_forward=_pivot_ilr_forward13,
                ilr_inverse=_pivot_ilr_inverse13,
                comp_to_xyz=_bary_to_xyz13,
                anchor_comp=None,
                vertices=_V13,
                plot_label=_source,
            )
            rng = np.random.default_rng(seed)
            sim_n = int(min(2000, max(300, X.shape[0]*sim_mult)))
            sim = rng.dirichlet(a, size=sim_n)
            xyz_sim = _bary_to_xyz13(sim)
            n_sim = int(xyz_sim.shape[0])
            ax.scatter(xyz_sim[:,0], xyz_sim[:,1], xyz_sim[:,2], s=point_size-2, alpha=0.22, marker='o', label=f"Dirichlet sim (n={n_sim})")
            return True, {"alpha": a, "S": _S, "n_obs": n_obs, "n_sim": n_sim, "source": _source}
        
        def _results_tetra13(block, sheet, labels, out_dir, title_extra="Observasjoner + Dirichlet(MLE)"):
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(6.2, 6))
            ax = fig.add_subplot(111, projection='3d')
            _plot_tetra13(ax, labels)
            ok, info = _overlay_observed_and_dirichlet13(ax, block, sheet, point_size=10)
            if ok and info:
                title_text = (f"{block}: {sheet} (source={info['source']}) — {title_extra}\n"
                              f"α={np.round(info['alpha'],3)}  S={float(info['S']):.3f}  "
                              f"n_obs={info['n_obs']}  n_sim={info['n_sim']}")
                ax.set_title(title_text)
                print(f"[TETRA TITLE] {title_text}")
            else:
                ax.set_title(f"{block}: {sheet} — {title_extra}")
            if ok: ax.legend(loc='best', fontsize=8, framealpha=0.6)
            fname = f"{_safe_name13(Path(xlsx_path).stem)}__{_safe_name13(block)}__{_safe_name13(sheet)}.png"
            out = Path(out_dir) / fname
            plt.tight_layout(); plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
            print(f"[STEG 13B] Lagret: {out}")
            register_output(step="STEG 13B", label="result_fig", path=out, kind="png")
        
        # ---------- 13.7: Fargelegg reelle data etter valgt ILR-koordinat ----------
        def _results_tetra_color_by_z13(block, sheet, labels, out_dir, color_by="z1", add_dirichlet_faint=False):
            out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            X = _empirical_matrix_for_sheet(sheet, block)
            if X is None or len(X) == 0:
                print(f"[STEG 13C] Ingen data for {block}/{sheet} – hopper over.")
                return
            Z   = _pivot_ilr_forward13(X)
            xyz = _bary_to_xyz13(X)
            fig = plt.figure(figsize=(6.2, 6))
            ax  = fig.add_subplot(111, projection='3d')
            _plot_tetra13(ax, labels)
            _source = _DIRICHLET_SOURCE_MAP.get((block, sheet), "UNKNOWN")
            _add_pivot_ilr_axes(
                ax=ax,
                ilr_forward=_pivot_ilr_forward13,
                ilr_inverse=_pivot_ilr_inverse13,
                comp_to_xyz=_bary_to_xyz13,
                anchor_comp=None,
                vertices=_V13,
                plot_label=_source,
            )
            if add_dirichlet_faint and not DIR_RES.empty:
                a, _S = _alpha_for_sheet(DIR_RES, block, sheet)
                if a is not None:
                    rng  = np.random.default_rng(123)
                    sim  = rng.dirichlet(a, size=min(1500, max(300, int(len(X)*1.5))))
                    xyzs = _bary_to_xyz13(sim)
                    n_sim = int(xyzs.shape[0])
                    ax.scatter(xyzs[:,0], xyzs[:,1], xyzs[:,2], s=8, alpha=0.16, marker='o', label=f"Dirichlet sim (n={n_sim})")
            idx = {"z1":0, "z2":1, "z3":2}[color_by]
            sc  = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=12, alpha=0.85, marker='o', c=Z[:,idx])
            cb  = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02); cb.set_label(f"{color_by} (pivot-ILR)")
            ax.set_title(f"{block}: {sheet} — fargekodet {color_by}")
            if add_dirichlet_faint:
                ax.legend(loc='best', fontsize=8, framealpha=0.6)
            out = Path(out_dir) / f"{_safe_name13(Path(xlsx_path).stem)}__{_safe_name13(block)}__{_safe_name13(sheet)}__color_by_{color_by}.png"
            plt.tight_layout(); plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
            print(f"[STEG 13C] Lagret: {out}")
            register_output(step="STEG 13C", label="result_fig", path=out, kind="png")
        
        # ---------- 13.8: Kjøring av figurer ----------
        _out_dir13 = Path(xlsx_path).with_name(Path(xlsx_path).stem + "_tetra_ilr")
        _out_dir13.mkdir(parents=True, exist_ok=True)
        
        # (A) Metode-demo (syntetisk) — OCAI-navn brukes for lesbarhet
        _methods_tetra_demo13(
            out_dir=_out_dir13,
            labels=["Klan","Adhockrati","Marked","Hierarki"],
            N=1000,
            alpha=(40,20,20,20),
            color_by="z1"
        )
        
        # (B) Reelle data (MERGET_*), hvis estimater finnes
        if not DIR_RES.empty:
            _results_tetra13("OCAI",     "MERGET_OCAI",     ocai_cols,  _out_dir13, "Pivot-ILR-akser")
            _results_tetra13("Strategi", "MERGET_STRATEGI", strat_cols, _out_dir13, "Pivot-ILR-akser")
        else:
            print("[STEG 13B] Hoppet over resultatsfigurer: DIR_RES mangler/er tom.")
        
        # (C) Reelle data — fargekodet av z1/z2/z3
        for block, sheet, labels in [
            ("OCAI",     "MERGET_OCAI",     ocai_cols),
            ("Strategi", "MERGET_STRATEGI", strat_cols),
        ]:
            for zlab in ("z1", "z2", "z3"):
                _results_tetra_color_by_z13(block, sheet, labels, _out_dir13, color_by=zlab, add_dirichlet_faint=False)
        
        
        # Konsoll-oversikt av ILR-reliabilitet (tallene som også ble skrevet til Excel)
        if not rel_O_ilr13.empty:
            _print_table("[STEG 13] OCAI – ILR reliabilitet",
                         rel_O_ilr13[["Profile","N_used","avg_inter_item_r_ilr",
                                       "alpha_from_r_ilr","split_half_SB_ilr"]])
        if not rel_S_ilr13.empty:
            _print_table("[STEG 13] Strategi – ILR reliabilitet",
                         rel_S_ilr13[["Profile","N_used","avg_inter_item_r_ilr",
                                       "alpha_from_r_ilr","split_half_SB_ilr"]])

    
        # Samler funksjoner for eksplisitte MANOVA-kall med sterk feilhåndtering/diagnostikk.
        # Forventet input er ILR-data + en faktorvariabel, og output er rapporttabeller som kan skrives til Excel.
        # ============================
        # === STEG 14: MANOVA (explicit df, strong diagnostics + actual runs)
        # ============================

        from typing import Dict, Optional, Union

        # MANOVA er valgfritt (gir melding hvis utilgjengelig)
        try:
            from statsmodels.multivariate.manova import MANOVA
            _HAS_MANOVA_14 = True
        except Exception:
            _HAS_MANOVA_14 = False
            print("[STEG 14] MANOVA ikke tilgjengelig (statsmodels). Hopper over MANOVA-kjøring.")

        def _coerce_for_patsy(df: pd.DataFrame, categorical_cols, numeric_cols) -> pd.DataFrame:
            """
            Ensure patsy/statsmodels-compatible dtypes:
              - categoricals => 'category'
              - numerics     => float (coerce errors to NaN)
            """
            out = df.copy()
            for c in categorical_cols:
                if c in out.columns:
                    out[c] = out[c].astype("category")
            for c in numeric_cols:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            return out

        def _extract_mv_tables(mv_res) -> Dict[str, Dict[str, pd.DataFrame]]:
            """
            Convert MANOVA mv_test() result into plain DataFrames.
            Returns {term: {test_name: DataFrame}}.
            """
            out: Dict[str, Dict[str, pd.DataFrame]] = {}
            for term, term_obj in mv_res.results.items():
                stat_df = term_obj["stat"].copy()
                out[term] = {"Tests": stat_df}
            return out

        def _write_mv_to_excel(
            mv_tables: dict,
            outfile: Optional[Union[str, Path]],
            meta: Optional[dict] = None,
            skip_tests_suffix: bool = False,
        ) -> None:
            """
            Write MANOVA tables to an Excel file if outfile is provided.
            Accepts str or pathlib.Path. No-op if outfile is None.
            """
            if outfile is None:
                return
            outpath = Path(outfile)
            if outpath.parent and not outpath.parent.exists():
                outpath.parent.mkdir(parents=True, exist_ok=True)

            with pd.ExcelWriter(outpath, engine="xlsxwriter") as xw:
                for term_tag, tbls in mv_tables.items():
                    for name, df in tbls.items():
                        # Add labeling columns to make tables self-explanatory.
                        # term_tag is expected like "OCAI_Departement_C(Departement)" from _merge_mv_tables14.
                        if meta and term_tag in meta:
                            block = meta[term_tag].get("block", "")
                            factor = meta[term_tag].get("factor", "")
                            term_key = meta[term_tag].get("term", term_tag)
                        else:
                            parts = term_tag.split("_", 2)
                            if len(parts) == 3:
                                block, factor, term_key = parts
                            else:
                                block, factor, term_key = "", "", term_tag
                        df_out = df.copy()
                        df_out.insert(0, "test", df_out.index.astype(str))
                        df_out = df_out.reset_index(drop=True)
                        df_out.insert(0, "term", term_key)
                        df_out.insert(0, "factor", factor)
                        df_out.insert(0, "block", block)
                        if skip_tests_suffix and name == "Tests":
                            sheet = f"{term_tag}".replace("/", "_")[:31]  # Excel limit 31
                        else:
                            sheet = f"{term_tag}_{name}".replace("/", "_")[:31]  # Excel limit 31
                        export_excel(df_out, writer=xw, sheet_name=sheet, label=sheet)

        def _manova_block_by_factor14(
            df: pd.DataFrame,
            block_name: str,
            prof_cols: list,
            factor_col: str,
            outfile: Optional[Union[str, Path]] = None,
            min_per_level: int = 2,
            min_levels: int = 2,
            min_n: int = 20,
        ):
            """
            Run MANOVA on ILR responses (3 dims) for a given block vs one factor.

            Parameters
            ----------
            df : pd.DataFrame
                Dataframe that *contains* the ILR columns and the factor column.
            block_name : str
                "OCAI" or "Strategi" (uses first letter to pick ilr columns).
            prof_cols : list
                Kept for signature compatibility; not used here.
            factor_col : str
                Categorical factor (e.g., "Departement").
            outfile : str | Path | None
                Optional Excel path for results.
            min_per_level : int
                Minimum observations per factor level to keep.
            min_levels : int
                Minimum number of levels required after filtering.
            min_n : int
                Minimum N required to attempt MANOVA.

            Returns
            -------
            dict with results metadata and tables.
            """
            if not _HAS_MANOVA_14:
                raise RuntimeError("[STEG 14] MANOVA ikke tilgjengelig (statsmodels).")

            # Determine expected ILR column names from block initial
            init = block_name[0]  # 'O' or 'S'
            ycols = [f'ilr{init}_1', f'ilr{init}_2', f'ilr{init}_3']

            # 1) Validate presence
            required = [factor_col] + ycols
            missing = [c for c in required if c not in df.columns]
            if missing:
                available_sample = ", ".join(list(df.columns)[:20])
                raise KeyError(
                    f"[MANOVA {block_name}] Missing columns: {missing}\n"
                    f"- Expected ILR columns: {ycols}\n"
                    f"- Factor column: '{factor_col}'\n"
                    f"- Available (first 20): {available_sample}"
                )

            # 2) Coerce dtypes (critical for 'string[python]' TypeError)
            df_work = _coerce_for_patsy(df, [factor_col], ycols)

            # 3) Drop rows with NA in factor or any response
            before_n = len(df_work)
            df_work = df_work.dropna(subset=[factor_col] + ycols)
            after_n = len(df_work)
            if df_work.empty:
                dtypes_info = df[required].dtypes.astype(str).to_dict()
                na_counts = df[required].isna().sum().to_dict()
                raise ValueError(
                    f"[MANOVA {block_name}] No rows left after dropping NAs.\n"
                    f"- Rows before: {before_n}, after: {after_n}\n"
                    f"- Dtypes: {dtypes_info}\n"
                    f"- NA counts: {na_counts}\n"
                    f"Tip: ensure ILR columns are numeric (float) and '{factor_col}' has no missing values."
                )

            # 3b) Filter factor levels with too few obs
            counts = df_work[factor_col].value_counts()
            keep_levels = counts.index[counts >= min_per_level]
            df_work = df_work[df_work[factor_col].isin(keep_levels)].copy()

            n_levels = int(df_work[factor_col].nunique())
            n_used = int(len(df_work))
            if n_levels < min_levels or n_used < min_n:
                raise ValueError(
                    f"[MANOVA {block_name}] Not enough data after level-filter.\n"
                    f"- Factor: {factor_col}\n"
                    f"- N_used: {n_used} (min_n={min_n})\n"
                    f"- Levels: {n_levels} (min_levels={min_levels})\n"
                    f"- min_per_level: {min_per_level}\n"
                    f"Tip: For numeric-like vars (e.g., Alder), consider binning before MANOVA if needed."
                )

            # 4) Build formula and fit
            formula = f"{' + '.join(ycols)} ~ C({factor_col})"
            ma = MANOVA.from_formula(formula, data=df_work)
            mv_res = ma.mv_test()

            # 5) Collect outputs
            mv_tables = _extract_mv_tables(mv_res)

            # 6) Write Excel (optional)
            _write_mv_to_excel(mv_tables, outfile)

            return {
                "block": block_name,
                "factor": factor_col,
                "n": n_used,
                "levels": n_levels,
                "ycols": ycols,
                "tables": mv_tables,
                "formula": formula,
                "outfile": str(outfile) if outfile is not None else None,
            }

        # ============================
        # 14.X: Faktiske kjøringer (hvis df finnes fra STEG 11)
        # ============================

        if _HAS_MANOVA_14:
            try:
                _df_ocai_14 = _ocai_ilr
            except NameError:
                _df_ocai_14 = None

            try:
                _df_strat_14 = _strat_ilr
            except NameError:
                _df_strat_14 = None

            _factor_candidates_14 = ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]

            # Samle alle tabeller og skriv én samlet fil (for å unngå overskriving)
            _ALL_MV_TABLES_14 = {}
            _RUN_META_14 = []

            def _merge_mv_tables14(block: str, factor: str, mv_tables: dict):
                # Prefix term-navn så det ikke kolliderer
                for term, tbls in mv_tables.items():
                    term2 = f"{block}_{factor}_{term}"
                    _ALL_MV_TABLES_14[term2] = tbls

            def _run_block14(df_block: pd.DataFrame, block_name: str):
                if df_block is None or getattr(df_block, "empty", True):
                    print(f"[STEG 14] Ingen df for {block_name} (mangler/er tom). Hopper over.")
                    return
                for factor in _factor_candidates_14:
                    if factor not in df_block.columns:
                        continue
                    try:
                        out = _manova_block_by_factor14(
                            df=df_block,
                            block_name=block_name,
                            prof_cols=(ocai_cols if block_name == "OCAI" else strat_cols),
                            factor_col=factor,
                            outfile=None,  # vi skriver samlet fil til slutt
                            min_per_level=2,
                            min_levels=2,
                            min_n=20,
                        )
                        _merge_mv_tables14(block_name, factor, out["tables"])
                        _RUN_META_14.append({
                            "block": out["block"],
                            "factor": out["factor"],
                            "n": out["n"],
                            "levels": out["levels"],
                            "formula": out["formula"],
                        })
                        print(f"[STEG 14] OK MANOVA {block_name} ~ {factor}: N={out['n']}, levels={out['levels']}")
                    except Exception as e:
                        print(f"[STEG 14] MANOVA hoppet over for {block_name} ~ {factor}: {repr(e)}")

            _run_block14(_df_ocai_14, "OCAI")
            _run_block14(_df_strat_14, "Strategi")

            def _pick_role_col(df_block: pd.DataFrame):
                for cand in ["Rolle", "Stilling", "role", "position"]:
                    if cand in df_block.columns:
                        return cand
                return None

            def _mv_tables_to_df(mv_tables: dict, block: str, factor: str) -> pd.DataFrame:
                rows = []
                for term, tbls in mv_tables.items():
                    stat_df = tbls.get("Tests")
                    if stat_df is None or getattr(stat_df, "empty", True):
                        continue
                    for test_name, row in stat_df.iterrows():
                        rowd = row.to_dict()
                        def _pick(keys):
                            for k in keys:
                                if k in rowd:
                                    return rowd.get(k)
                            return None
                        rows.append({
                            "block": block,
                            "factor": factor,
                            "group": term,
                            "test": str(test_name),
                            "stat": _pick(["Value", "value", "Stat", "stat"]),
                            "F": _pick(["F Value", "F", "f"]),
                            "df1": _pick(["Num DF", "num df", "df1", "DF1"]),
                            "df2": _pick(["Den DF", "den df", "df2", "DF2"]),
                            "p": _pick(["Pr > F", "Pr>F", "p", "p_value"]),
                        })
                if not rows:
                    return pd.DataFrame()
                return pd.DataFrame(rows)

            _role_rows = []
            for _block_name, _df_block in [("OCAI", _df_ocai_14), ("Strategi", _df_strat_14)]:
                if _df_block is None or getattr(_df_block, "empty", True):
                    continue
                _role_col = _pick_role_col(_df_block)
                if _role_col is None:
                    continue
                _counts = _df_block[_role_col].value_counts(dropna=True)
                for _level, _count in _counts.items():
                    print(f"[STEG 14] ROLE_COUNTS block={_block_name} factor={_role_col} level={_level} n={int(_count)}")
                try:
                    _out_role = _manova_block_by_factor14(
                        df=_df_block,
                        block_name=_block_name,
                        prof_cols=(ocai_cols if _block_name == "OCAI" else strat_cols),
                        factor_col=_role_col,
                        outfile=None,
                        min_per_level=2,
                        min_levels=2,
                        min_n=20,
                    )
                    _role_df = _mv_tables_to_df(_out_role["tables"], _block_name, _role_col)
                    if not _role_df.empty:
                        _role_rows.append(_role_df)
                except Exception as e:
                    print(f"[STEG 14] MANOVA role/stilling hoppet over for {_block_name} ~ {_role_col}: {repr(e)}")

            if _role_rows:
                MANOVA_ROLE_DF = pd.concat(_role_rows, axis=0, ignore_index=True)
                _log_p_allrows(MANOVA_ROLE_DF, "STEG14", "MANOVA_ROLE_ALL", cap=200)
                _log_sig_summary(MANOVA_ROLE_DF.rename(columns={"test": "term"}), "MANOVA_ROLE_SIG", "STEG14", cap=200)

            if _ALL_MV_TABLES_14:
                _out_manova14 = base_path.with_name(base_path.stem + "_manova_explicit.xlsx") if "base_path" in globals() else Path(xlsx_path).with_name(Path(xlsx_path).stem + "_manova_explicit.xlsx")
                _write_mv_to_excel(_ALL_MV_TABLES_14, _out_manova14)

                # Meta-ark i samme fil (praktisk)
                try:
                    with pd.ExcelWriter(_out_manova14, engine="xlsxwriter", mode="a", if_sheet_exists="replace") as xw:
                        meta = pd.DataFrame(_RUN_META_14)
                        export_excel(meta, writer=xw, sheet_name="RUN_META", label="RUN_META")
                except Exception:
                    # Hvis xlsxwriter ikke støtter append i miljøet, logges meta til konsoll
                    pass

                print(f"[STEG 14] Skrev MANOVA-tabeller til: {_out_manova14}")
                register_output(step="STEG 14", label="manova_tables", path=_out_manova14, kind="xlsx")
            if _RUN_META_14:
                print("[STEG 14] RUN_META (meta):")
                _log_df_meta("[STEG 14] RUN_META", pd.DataFrame(_RUN_META_14))
            else:
                print("[STEG 14] Ingen MANOVA-resultater å skrive (ingen kjøringer lyktes / nok data).")

            # ---- 14.L: Likert MANOVA (Kontroll) per faktor ----------------------
            _ALL_MV_TABLES_LIKERT = {}
            _RUN_META_LIKERT = []
            _LIKERT_META = {}

            # Load Kontroll + Bakgrunn directly (ID + bg vars + Likert cols)
            _likert_df14 = pd.DataFrame()
            _likert_cols14 = []
            _likert_bg_cols14 = []
            try:
                _xls14 = pd.ExcelFile(xlsx_path)
                _sh_ctrl14 = find_sheet_name("Kontroll", _xls14.sheet_names)
                _sh_bg14 = (find_sheet_name("Bakgrunn", _xls14.sheet_names)
                            or find_sheet_name("Background", _xls14.sheet_names))
                if _sh_ctrl14 is not None:
                    _ctrl14 = pd.read_excel(xlsx_path, sheet_name=_sh_ctrl14)
                    _ctrl14.columns = _ctrl14.columns.astype(str).str.strip()
                    if "ID" in _ctrl14.columns:
                        _ctrl14 = _ctrl14.copy()
                        _ctrl14["ID"] = _ctrl14["ID"].astype(str)
                        _likert_cols14 = [c for c in _ctrl14.columns if c != "ID"]
                        if _sh_bg14 is not None:
                            _bg14 = pd.read_excel(xlsx_path, sheet_name=_sh_bg14)
                            _bg14.columns = _bg14.columns.astype(str).str.strip()
                            if "ID" in _bg14.columns:
                                _bg14 = _bg14.copy()
                                _bg14["ID"] = _bg14["ID"].astype(str)
                                _likert_bg_cols14 = [c for c in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"] if c in _bg14.columns]
                                _bg_keep14 = _bg14[["ID"] + _likert_bg_cols14].drop_duplicates("ID")
                                _likert_df14 = _bg_keep14.merge(_ctrl14, on="ID", how="left")
                            else:
                                _likert_df14 = _ctrl14
                        else:
                            _likert_df14 = _ctrl14
            except Exception:
                _likert_df14 = pd.DataFrame()
                _likert_cols14 = []
                _likert_bg_cols14 = []

            if _likert_df14 is not None and not _likert_df14.empty:
                _likert_df14.columns = _likert_df14.columns.astype(str).str.strip()
            _present_factors = [c for c in ["Departement", "Ansiennitet", "Alder", "Kjønn", "Stilling"]
                                if _likert_df14 is not None and c in _likert_df14.columns]
            _shape = _likert_df14.shape if _likert_df14 is not None else (0, 0)
            print(f"[LIKERT_MANOVA] columns present: {_present_factors} | merged shape={_shape}")
            _likert_col_map = {orig: f"Y{i+1}" for i, orig in enumerate(_likert_cols14)}
            _likert_col_map_df = pd.DataFrame(
                [{"original_col": k, "safe_col": v} for k, v in _likert_col_map.items()]
            )

            def _merge_mv_tables_likert(factor: str, mv_tables: dict):
                _abbr_map = {
                    "Departement": "DEP",
                    "Ansiennitet": "ANS",
                    "Alder": "ALD",
                    "Kjønn": "KJ",
                    "Stilling": "STI",
                }
                abbr = _abbr_map.get(factor, factor[:3].upper())
                for term, tbls in mv_tables.items():
                    tag = "INT" if term == "Intercept" else "C"
                    term2 = f"L_{abbr}_{tag}_Tests"
                    _LIKERT_META[term2] = {"block": "LIKERT", "factor": factor, "term": term}
                    _ALL_MV_TABLES_LIKERT[term2] = tbls

            def _run_likert_factor(factor: str):
                if _likert_df14 is None or _likert_df14.empty or not _likert_cols14:
                    _RUN_META_LIKERT.append({
                        "block": "LIKERT",
                        "factor": factor,
                        "n_total_before_dropna": np.nan,
                        "n_used_after_dropna": np.nan,
                        "k_groups_used": np.nan,
                        "min_group_n": np.nan,
                        "formula": "",
                        "skip_reason": "Likert data missing/empty",
                    })
                    return
                if factor not in _likert_df14.columns:
                    _RUN_META_LIKERT.append({
                        "block": "LIKERT",
                        "factor": factor,
                        "n_total_before_dropna": np.nan,
                        "n_used_after_dropna": np.nan,
                        "k_groups_used": np.nan,
                        "min_group_n": np.nan,
                        "formula": "",
                        "skip_reason": "factor missing",
                    })
                    return
                try:
                    df_model = _likert_df14.copy()
                    if _likert_col_map:
                        df_model = df_model.rename(columns=_likert_col_map)
                    safe_y_cols = list(_likert_col_map.values())
                    df_work = _coerce_for_patsy(df_model, [factor], safe_y_cols)
                    n_total = int(len(df_work))
                    df_work = df_work.dropna(subset=[factor] + safe_y_cols)
                    n_used = int(len(df_work))
                    counts = df_work[factor].value_counts()
                    min_group_n = int(counts.min()) if not counts.empty else 0
                    keep_levels = counts.index[counts >= 2]
                    df_work = df_work[df_work[factor].isin(keep_levels)].copy()
                    n_levels = int(df_work[factor].nunique())
                    if n_levels < 2 or n_used < 20:
                        raise ValueError(f"Not enough data after level-filter (n={n_used}, levels={n_levels})")
                    formula = f"{' + '.join(safe_y_cols)} ~ C(Q(\"{factor}\"))"
                    ma = MANOVA.from_formula(formula, data=df_work)
                    mv_res = ma.mv_test()
                    mv_tables = _extract_mv_tables(mv_res)
                    # keep only Intercept + C(factor) terms if present
                    keep_terms = {}
                    want_term = f"C(Q(\"{factor}\"))"
                    for term_key in mv_tables.keys():
                        if term_key == "Intercept" or term_key == want_term:
                            keep_terms[term_key] = mv_tables[term_key]
                    _merge_mv_tables_likert(factor, keep_terms if keep_terms else mv_tables)
                    _RUN_META_LIKERT.append({
                        "block": "LIKERT",
                        "factor": factor,
                        "n_total_before_dropna": n_total,
                        "n_used_after_dropna": n_used,
                        "k_groups_used": n_levels,
                        "min_group_n": min_group_n,
                        "formula": formula,
                        "skip_reason": "",
                    })
                except Exception as e:
                    _RUN_META_LIKERT.append({
                        "block": "LIKERT",
                        "factor": factor,
                        "n_total_before_dropna": np.nan,
                        "n_used_after_dropna": np.nan,
                        "k_groups_used": np.nan,
                        "min_group_n": np.nan,
                        "formula": formula if "formula" in locals() else "",
                        "skip_reason": repr(e),
                    })

            for factor in _factor_candidates_14:
                _run_likert_factor(factor)

            _out_likert_manova14 = base_path.with_name(base_path.stem + "_likert_manova_explicit.xlsx") if "base_path" in globals() else Path(xlsx_path).with_name(Path(xlsx_path).stem + "_likert_manova_explicit.xlsx")
            if _ALL_MV_TABLES_LIKERT:
                _write_mv_to_excel(_ALL_MV_TABLES_LIKERT, _out_likert_manova14, meta=_LIKERT_META, skip_tests_suffix=True)
            else:
                # still create workbook with RUN_META only
                with pd.ExcelWriter(_out_likert_manova14, engine="xlsxwriter") as xw:
                    meta = pd.DataFrame(_RUN_META_LIKERT)
                    export_excel(meta, writer=xw, sheet_name="RUN_META", label="RUN_META")
                    export_excel(_likert_col_map_df, writer=xw, sheet_name="COL_MAP", label="COL_MAP")
            try:
                with pd.ExcelWriter(_out_likert_manova14, engine="openpyxl", mode="a", if_sheet_exists="replace") as xw:
                    meta = pd.DataFrame(_RUN_META_LIKERT)
                    export_excel(meta, writer=xw, sheet_name="RUN_META", label="RUN_META")
                    export_excel(_likert_col_map_df, writer=xw, sheet_name="COL_MAP", label="COL_MAP")
            except Exception:
                pass
            if _ALL_MV_TABLES_LIKERT:
                print(f"[STEG 14] Skrev LIKERT MANOVA-tabeller til: {_out_likert_manova14}")
            else:
                print(f"[STEG 14] Skrev LIKERT RUN_META (ingen tester) til: {_out_likert_manova14}")
            register_output(step="STEG 14", label="likert_manova_tables", path=_out_likert_manova14, kind="xlsx")


    
    
    # Bygger ILR-profiler per ID og korrelerer Strategi-balansene mot OCAI og mot kontroller (Likert/numerisk bakgrunn).
    # Input er tidligere _merged_profiles_with_id og Kontrol-data; output er Spearman-matriser i Excel og konsollutdrag.
    # ============================
    # STEG 15: Spearman på ILR-balanser (sensitivitet for H5–H6)
    # ============================
    # Forutsetter:
    # - ocai_cols, strat_cols, xlsx_path, base_path
    # - _merged_profiles_with_id (fra tidligere steg; ID + 4 profiler i 0..1)
    # - _load_kontrol_sheet (fra tidligere steg; ID + A..E (+ ev. numerisk bg))
    # Endrer ikke eksisterende datastrukturer; nye hjelpere får suffiks "15".
    
    def _ilr_pivot_from_matrix15(P4, eps=1e-12):
        """
        P4: ndarray (n,4) med rader som summerer ~1.
        Returnerer Z (n,3) = pivot-ILR for orden (p1,p2,p3,p4):
          z1 = p1 vs {p2,p3,p4}, z2 = p2 vs {p3,p4}, z3 = p3 vs p4.
        """
        X = np.array(P4, dtype=float, copy=True)
        X[X <= 0] = eps
        X = X / X.sum(axis=1, keepdims=True)
        L = np.log(X)
        clr = L - L.mean(axis=1, keepdims=True)
        PSI = np.array([
            [ np.sqrt(3/4),         0.0,                 0.0               ],
            [-1/np.sqrt(12),  np.sqrt(2/3),              0.0               ],
            [-1/np.sqrt(12), -1/np.sqrt(6),        1/np.sqrt(2)           ],
            [-1/np.sqrt(12), -1/np.sqrt(6),       -1/np.sqrt(2)           ],
        ])
        return clr @ PSI  # (n,3)
    
    def _build_ilr_with_id15(block_name, profile_cols):
        """
        Bygger per-ID ILR (pivot) for angitt blokk ('OCAI'/'Strategi')
        via _merged_profiles_with_id (ID + 4 profiler i 0..1).
        Returnerer DF: ['ID','ilrX_1','ilrX_2','ilrX_3'] hvor X = 'O' eller 'S'.
        """
        M = _merged_profiles_with_id(block_name)
        if M.empty:
            return pd.DataFrame()
    
        have = [c for c in profile_cols if c in M.columns]
        if len(have) != 4:
            return pd.DataFrame()
    
        P = M[have].to_numpy(float)
        sums = P.sum(axis=1)
        keep = np.isfinite(sums) & (np.abs(sums - 1.0) <= 1e-6)
        M = M.loc[keep].reset_index(drop=True)
        P = P[keep]
    
        if P.size == 0:
            return pd.DataFrame()
    
        Z = _ilr_pivot_from_matrix15(P)  # (n,3)
        pre = 'O' if block_name == 'OCAI' else 'S'
        out = pd.DataFrame({
            "ID": M["ID"].values,
            f"ilr{pre}_1": Z[:,0],
            f"ilr{pre}_2": Z[:,1],
            f"ilr{pre}_3": Z[:,2],
        })
        return out
    
    def _numeric_controls_df15():
        """
        Leser 'Kontrol(l)'-arket via _load_kontrol_sheet og returnerer DF med:
        ['ID'] + A..E + ev. numeriske bakgrunnsvariabler (Alder/Ansiennitet).
        Ikke-numeriske felt droppes for Spearman.
        """
        K = _load_kontrol_sheet()
        if K.empty or "ID" not in K.columns:
            return pd.DataFrame()
    
        out = K[["ID"]].copy()
    
        # Likert A..E (behold de som finnes)
        for c in ["A","B","C","D","E"]:
            if c in K.columns:
                out[c] = pd.to_numeric(K[c], errors='coerce')
    
        # Ev. numerisk bakgrunn
        for c in ["Alder","Ansiennitet"]:
            if c in K.columns:
                out[c] = pd.to_numeric(K[c], errors='coerce')
    
        keep_cols = ["ID"] + [c for c in out.columns if c != "ID" and out[c].notna().any()]
        return out[keep_cols] if len(keep_cols) > 1 else pd.DataFrame()
    
    def _spearman_block15(X_cols, Y_cols, DF):
        """
        Returnerer (rho_df, p_df) for Spearman mellom X- og Y-kolonner i DF.
        """
        cols = X_cols + Y_cols
        D = DF[cols].copy()
        mask_any = D[cols].notna().any(axis=1)
        D = D.loc[mask_any]
        if D.empty:
            return pd.DataFrame(), pd.DataFrame()
        rho, p = spearmanr(D[cols], axis=0, nan_policy='omit')
        kx = len(X_cols); ky = len(Y_cols)
        rho_block = pd.DataFrame(rho[:kx, kx:kx+ky], index=X_cols, columns=Y_cols)
        p_block   = pd.DataFrame(p[:kx,   kx:kx+ky], index=X_cols, columns=Y_cols)
        return rho_block, p_block
    
    # --- Bygg datasett: ILR(OCAI), ILR(Strategi), Kontroller (Likert + numerisk bg)
    _ilr_O_15 = _build_ilr_with_id15("OCAI", ocai_cols)
    _ilr_S_15 = _build_ilr_with_id15("Strategi", strat_cols)
    _ctrl_15  = _numeric_controls_df15()
    
    # Flett sammen på ID (inner join for parvis sammenlignbarhet)
    frames_15 = [df for df in [_ilr_S_15, _ilr_O_15, _ctrl_15] if not df.empty]
    if len(frames_15) >= 2:
        M_15 = frames_15[0]
        for df in frames_15[1:]:
            M_15 = M_15.merge(df, on="ID", how="inner")
    else:
        M_15 = pd.DataFrame()
    
    # --- H5: Spearman( ilrS •, ilrO • )
    if not M_15.empty:
        Scols_15 = [c for c in ["ilrS_1","ilrS_2","ilrS_3"] if c in M_15.columns]
        Ocols_15 = [c for c in ["ilrO_1","ilrO_2","ilrO_3"] if c in M_15.columns]
        if Scols_15 and Ocols_15:
            rho_SO_15, p_SO_15 = _spearman_block15(Scols_15, Ocols_15, M_15)
        else:
            rho_SO_15, p_SO_15 = pd.DataFrame(), pd.DataFrame()
    else:
        rho_SO_15, p_SO_15 = pd.DataFrame(), pd.DataFrame()
        Scols_15, Ocols_15 = [], []
    
    # --- H6: Spearman( ilrS •, kontroller A–E + numerisk bg )
    if not M_15.empty:
        Ccols_15 = [c for c in ["A","B","C","D","E","Alder","Ansiennitet"] if c in M_15.columns]
        if Scols_15 and Ccols_15:
            rho_SC_15, p_SC_15 = _spearman_block15(Scols_15, Ccols_15, M_15)
        else:
            rho_SC_15, p_SC_15 = pd.DataFrame(), pd.DataFrame()
    else:
        rho_SC_15, p_SC_15 = pd.DataFrame(), pd.DataFrame()
        Ccols_15 = []
    
    # --- Skriv til Excel (samme filnavn som tidligere implementasjon for kompatibilitet)
    out_corr_15 = xlsx_path.with_name(xlsx_path.stem + "_spearman_ILR_sensitivity.xlsx")
    with pd.ExcelWriter(out_corr_15, engine="xlsxwriter") as w:
        if not rho_SO_15.empty:
            export_excel(rho_SO_15, writer=w, sheet_name="H5_S_vs_O_rho", label="H5_S_vs_O_rho")
            export_excel(p_SO_15, writer=w, sheet_name="H5_S_vs_O_p", label="H5_S_vs_O_p")
            long_H5 = _spearman_matrix_to_long(rho_SO_15, p_SO_15, "Strategi_vs_OCAI", "ILR", "spearman", len(M_15) if not M_15.empty else None)
            if not long_H5.empty:
                export_excel(long_H5, writer=w, sheet_name="H5_LONG", label="H5_LONG")
                _log_sig_summary(long_H5, "H5_LONG", "STEG15")
        if not rho_SC_15.empty:
            export_excel(rho_SC_15, writer=w, sheet_name="H6_S_vs_Ctrl_rho", label="H6_S_vs_Ctrl_rho")
            export_excel(p_SC_15, writer=w, sheet_name="H6_S_vs_Ctrl_p", label="H6_S_vs_Ctrl_p")
            long_H6 = _spearman_matrix_to_long(rho_SC_15, p_SC_15, "Strategi_vs_Kontroll", "ILR", "spearman", len(M_15) if not M_15.empty else None)
            if not long_H6.empty:
                export_excel(long_H6, writer=w, sheet_name="H6_LONG", label="H6_LONG")
                _log_sig_summary(long_H6, "H6_LONG", "STEG15")
        if not M_15.empty:
            keep_cols_15 = ["ID"] + Scols_15 + Ocols_15 + Ccols_15
            export_excel(pd.DataFrame(M_15[keep_cols_15]), writer=w, sheet_name="Data_used", label="Data_used")

        meta_rows_15 = []
        if not rho_SO_15.empty:
            meta_rows_15.extend([
                {"outfile": out_corr_15.name, "sheet_name": "H5_S_vs_O_rho", "block": "Strategi_vs_OCAI", "space": "ILR", "method": "Spearman", "variables": "ilrS vs ilrO", "n_used": len(M_15) if not M_15.empty else ""},
                {"outfile": out_corr_15.name, "sheet_name": "H5_S_vs_O_p", "block": "Strategi_vs_OCAI", "space": "ILR", "method": "Spearman", "variables": "ilrS vs ilrO (p)", "n_used": len(M_15) if not M_15.empty else ""},
                {"outfile": out_corr_15.name, "sheet_name": "H5_LONG", "block": "Strategi_vs_OCAI", "space": "ILR", "method": "Spearman", "variables": "ilrS vs ilrO (long)", "n_used": len(M_15) if not M_15.empty else ""},
            ])
        if not rho_SC_15.empty:
            meta_rows_15.extend([
                {"outfile": out_corr_15.name, "sheet_name": "H6_S_vs_Ctrl_rho", "block": "Strategi_vs_Kontroll", "space": "ILR", "method": "Spearman", "variables": "ilrS vs kontroll", "n_used": len(M_15) if not M_15.empty else ""},
                {"outfile": out_corr_15.name, "sheet_name": "H6_S_vs_Ctrl_p", "block": "Strategi_vs_Kontroll", "space": "ILR", "method": "Spearman", "variables": "ilrS vs kontroll (p)", "n_used": len(M_15) if not M_15.empty else ""},
                {"outfile": out_corr_15.name, "sheet_name": "H6_LONG", "block": "Strategi_vs_Kontroll", "space": "ILR", "method": "Spearman", "variables": "ilrS vs kontroll (long)", "n_used": len(M_15) if not M_15.empty else ""},
            ])
        if not M_15.empty:
            meta_rows_15.append({"outfile": out_corr_15.name, "sheet_name": "Data_used", "block": "Strategi/OCAI/Kontroll", "space": "ILR", "method": "Data", "variables": "ID + ilr + controls", "n_used": len(M_15)})
        export_excel(pd.DataFrame(meta_rows_15), writer=w, sheet_name="RUN_META", label="RUN_META")
    
    print("[STEG 15 / SPEARMAN-ILR] Skrev sensitivitetskorrelasjoner til:", out_corr_15)
    register_output(step="STEG 15", label="spearman_ilr", path=out_corr_15, kind="xlsx")
    if not rho_SO_15.empty:
        print("\n[H5] Spearman (ilrS vs ilrO) – rho:")
        _log_df_meta("[H5] rho_SO_15 (meta)", rho_SO_15)
    if not rho_SC_15.empty:
        print("\n[H6] Spearman (ilrS vs Kontroller) – rho:")
        _log_df_meta("[H6] rho_SC_15 (meta)", rho_SC_15)
    
    
    # Viser pivot-ILR-geometri i tetraederform: metodedemo på syntetiske data og visualiseringer av MERGET-data med/uten Dirichlet-sky.
    # Input er DIR_RES/p_tables og profilnavn; output er PNG-filer i _tetra_ilr-mappen.
    # ============================
    # STEG 16: TETRAHEDER – (A) METODE + (B) RESULTAT
    # ============================
    # Forutsetter fra tidligere steg:
    # - xlsx_path, ocai_cols, strat_cols
    # - DIR_RES (fra STEG 7), p_tables (fra STEG 2)
    # - _empirical_matrix_for_sheet, _alpha_for_sheet, _safe_name (fra STEG 8)
    # Endrer ikke eksisterende navn; alt nytt suffikses med "16".
    
    # ---------- Geometri (samme innbygging som tidligere), isolert for STEG 16 ----------
    def _tetra_vertices16():
        v1 = np.array([ 1.0,  0.0,  0.0])
        v2 = np.array([-1/3,  2*np.sqrt(2)/3,  0.0])
        v3 = np.array([-1/3, -np.sqrt(2)/3,  np.sqrt(6)/3])
        v4 = np.array([-1/3, -np.sqrt(2)/3, -np.sqrt(6)/3])
        return np.vstack([v1, v2, v3, v4])
    
    _V16 = _tetra_vertices16()
    _centroid16 = _V16.mean(axis=0)
    
    def _draw_tetra_wire16(ax):
        edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        for i,j in edges:
            ax.plot([_V16[i,0],_V16[j,0]], [_V16[i,1],_V16[j,1]], [_V16[i,2],_V16[j,2]], linewidth=1, alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    
    def _plot_tetra_with_axes16(ax, labels):
        _draw_tetra_wire16(ax)
        for i, lab in enumerate(labels):
            ax.scatter(_V16[i,0], _V16[i,1], _V16[i,2], s=120, marker='o', alpha=0.9)
            ax.text(_V16[i,0], _V16[i,1], _V16[i,2], "  "+lab, fontsize=9, ha='left', va='bottom')
    
    # ---------- Barysentriske hjelpetjenester ----------
    def _bary_to_xyz16(P_4col: np.ndarray) -> np.ndarray:
        """(n,4) med rader som summerer til 1 -> (n,3) xyz."""
        return P_4col @ _V16
    
    def _color_by_scalar16(ax, xyz, z, label):
        sc = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=10, alpha=0.7, marker='o', c=z)
        cb = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cb.set_label(label)
    
    # ---------- Pivot ILR: forward/inverse (identisk funksjonelt med tidligere) ----------
    def _pivot_ilr_forward16(P):
        """P: (n,4) sum=1 -> Z: (n,3) pivot-ILR (p1 vs rest; p2 vs {3,4}; p3 vs p4)."""
        P = np.asarray(P, float)
        P = np.clip(P, 1e-15, 1.0)
        p1,p2,p3,p4 = P[:,0], P[:,1], P[:,2], P[:,3]
        z1 = np.sqrt(3/4.0) * (np.log(p1) - (np.log(p2)+np.log(p3)+np.log(p4))/3.0)
        z2 = np.sqrt(2/3.0) * (np.log(p2) - (np.log(p3)+np.log(p4))/2.0)
        z3 = np.sqrt(1/2.0) * (np.log(p3) - np.log(p4))
        return np.column_stack([z1,z2,z3])
    
    def _pivot_ilr_inverse16(Z):
        """Z: (...,3) -> P: (...,4) for pivot-basis (p1,p2,p3,p4)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        a = np.exp(Z[:,0] * (2/np.sqrt(3)))   # exp(z1 / sqrt(3/4))
        b = np.exp(Z[:,1] * np.sqrt(3/2))     # exp(z2 / sqrt(2/3))
        c = np.exp(Z[:,2] * np.sqrt(2))       # exp(z3 / sqrt(1/2))
        p1_prop = a * (b**(1/3)) * (c**(1/2))
        p2_prop = b * (c**(1/2))
        p3_prop = c
        p4_prop = np.ones_like(c)
        P = np.stack([p1_prop, p2_prop, p3_prop, p4_prop], axis=1)
        P = np.clip(P, 1e-15, np.inf)
        return P / P.sum(axis=1, keepdims=True)
    
    def _aitchison_center_rows16(P):
        eps = 1e-15
        X = np.clip(np.asarray(P, float), eps, None)
        X = X / X.sum(axis=1, keepdims=True)
        logX = np.log(X)
        m = np.mean(logX, axis=0)
        p0 = np.exp(m)
        p0 = p0 / p0.sum()
        return p0
    
    def _draw_ilr_arrows16(ax, P_obs, source_label):
        if P_obs is None or len(P_obs) == 0:
            return
        p0 = _aitchison_center_rows16(P_obs)
        z0 = _pivot_ilr_forward16(p0.reshape(1, -1))[0]
        Z = _pivot_ilr_forward16(P_obs)
        stds = np.std(Z, axis=0, ddof=1)
        deltas = []
        arrow_vecs = []
        xyz0 = _bary_to_xyz16(p0.reshape(1, -1))[0]
        for i in range(3):
            d = float(0.2 * stds[i]) if np.isfinite(stds[i]) and stds[i] > 0 else 0.5
            deltas.append(d)
            z1 = z0.copy()
            z1[i] += d
            p1 = _pivot_ilr_inverse16(z1.reshape(1, -1))[0]
            xyz1 = _bary_to_xyz16(p1.reshape(1, -1))[0]
            v = xyz1 - xyz0
            arrow_vecs.append(v.tolist())
            ax.quiver(xyz0[0], xyz0[1], xyz0[2], v[0], v[1], v[2],
                      arrow_length_ratio=0.08, linewidth=2)
            ax.text(xyz1[0], xyz1[1], xyz1[2], f"+z{i+1}", fontsize=10)
        print(f"[ILR ARROW] source={source_label} delta={deltas} anchor_xyz={xyz0.tolist()} arrow_vecs={arrow_vecs}")
    
    # ---------- Overlays på reelle data / Dirichlet (bruker STEG 8-hjelpere) ----------
    def _overlay_observed_and_dirichlet16(ax, block, sheet, point_size=10, sim_mult=2, seed=123):
        X = _empirical_matrix_for_sheet(sheet, block)   # observed P (n,4)
        a, _S = _alpha_for_sheet(DIR_RES, block, sheet) # fitted alpha (4,)
        if X is None or a is None:
            return False, None
        xyz_emp = _bary_to_xyz16(X)
        n_obs = int(xyz_emp.shape[0])
        ax.scatter(xyz_emp[:,0], xyz_emp[:,1], xyz_emp[:,2], s=point_size, alpha=0.45, marker='o', label=f"Observed (n={n_obs})")
        rng = np.random.default_rng(seed)
        sim_n = int(min(2000, max(300, X.shape[0]*sim_mult)))
        _source = _DIRICHLET_SOURCE_MAP.get((block, sheet), "UNKNOWN")
        _add_pivot_ilr_axes(
            ax=ax,
            ilr_forward=_pivot_ilr_forward16,
            ilr_inverse=_pivot_ilr_inverse16,
            comp_to_xyz=_bary_to_xyz16,
            anchor_comp=None,
            vertices=_V16,
            plot_label=_source,
        )
        print(f"[DIRICHLET SIM] block={block} source={_source} n_sim={sim_n}")
        _DIRICHLET_SIM_META.append({
            "block": block,
            "source": _source,
            "sheet_name": sheet,
            "n_sim": sim_n,
            "plot_step": "STEG 16",
        })
        sim = rng.dirichlet(a, size=sim_n)
        xyz_sim = _bary_to_xyz16(sim)
        n_sim = int(xyz_sim.shape[0])
        ax.scatter(xyz_sim[:,0], xyz_sim[:,1], xyz_sim[:,2], s=point_size-2, alpha=0.22, marker='o', label=f"Dirichlet sim (n={n_sim})")
        return True, {"alpha": a, "S": _S, "n_obs": n_obs, "n_sim": n_sim, "source": _source}
    
    # ---------- (A) Metode-figur: syntetisk Dirichlet ----------
    def _methods_tetra_demo16(out_dir, labels, N=800, alpha=(40,20,20,20), color_by="z1"):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(42)
        A = np.array(alpha, dtype=float)
        P = rng.dirichlet(A, size=N)             # synthetic compositions
        Z = _pivot_ilr_forward16(P)              # ILR-koordinater
        xyz = _bary_to_xyz16(P)                  # tetra-koordinater
    
        fig = plt.figure(figsize=(6.2, 6))
        ax = fig.add_subplot(111, projection='3d')
        _plot_tetra_with_axes16(ax, labels)
        _add_pivot_ilr_axes(
            ax=ax,
            ilr_forward=_pivot_ilr_forward16,
            ilr_inverse=_pivot_ilr_inverse16,
            comp_to_xyz=_bary_to_xyz16,
            anchor_comp=None,
            vertices=_V16,
            plot_label="DEMO16",
        )
        idx = {"z1":0, "z2":1, "z3":2}.get(color_by, 0)
        _color_by_scalar16(ax, xyz, Z[:,idx], label=f"{color_by} (pivot-ILR)")
        ax.set_title(f"Metode-demo: Dirichlet(alpha={list(alpha)}), pivot-ILR-akser")
        out = out_dir / "metode_synthetic_tetra_pivotILR.png"
        plt.tight_layout(); plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
        print(f"[STEG 16A] Lagret: {out}")
        register_output(step="STEG 16A", label="result_fig", path=out, kind="png")
    
    # ---------- (B) Resultat-figurer: MERGET_OCAI og MERGET_STRATEGI ----------
    def _results_tetra16(block, sheet, labels, out_dir, title_extra="Observasjoner + Dirichlet(MLE)"):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(6.2, 6))
        ax = fig.add_subplot(111, projection='3d')
        _plot_tetra_with_axes16(ax, labels)
        ok, info = _overlay_observed_and_dirichlet16(ax, block, sheet, point_size=10)
        if ok and info:
            title_text = (f"{block}: {sheet} (source={info['source']}) — {title_extra}\n"
                          f"α={np.round(info['alpha'],3)}  S={float(info['S']):.3f}  "
                          f"n_obs={info['n_obs']}  n_sim={info['n_sim']}")
            ax.set_title(title_text)
            print(f"[TETRA TITLE] {title_text}")
        else:
            ax.set_title(f"{block}: {sheet} — {title_extra}")
        if ok: ax.legend(loc='best', fontsize=8, framealpha=0.6)
        fname = f"{_safe_name(Path(xlsx_path).stem)}__{_safe_name(block)}__{_safe_name(sheet)}.png"
        out = Path(out_dir) / fname
        plt.tight_layout(); plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
        print(f"[STEG 16B] Lagret: {out}")
        register_output(step="STEG 16B", label="result_fig", path=out, kind="png")
    
    # ---------- (C) Real-data fargelagt etter ILR (z1/z2/z3) ----------
    def _results_tetra_color_by_z16(block, sheet, labels, out_dir, color_by="z1", add_dirichlet_faint=False):
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        X = _empirical_matrix_for_sheet(sheet, block)   # (n,4), sum≈1
        if X is None or len(X) == 0:
            print(f"[STEG 16C] Ingen data for {block}/{sheet} – hopper over.")
            return
    
        Z   = _pivot_ilr_forward16(X)
        xyz = _bary_to_xyz16(X)
    
        fig = plt.figure(figsize=(6.2, 6))
        ax  = fig.add_subplot(111, projection='3d')
        _plot_tetra_with_axes16(ax, labels)
        _source = _DIRICHLET_SOURCE_MAP.get((block, sheet), "UNKNOWN")
        _add_pivot_ilr_axes(
            ax=ax,
            ilr_forward=_pivot_ilr_forward16,
            ilr_inverse=_pivot_ilr_inverse16,
            comp_to_xyz=_bary_to_xyz16,
            anchor_comp=None,
            vertices=_V16,
            plot_label=_source,
        )
    
        if add_dirichlet_faint and not DIR_RES.empty:
            a, _S = _alpha_for_sheet(DIR_RES, block, sheet)
            if a is not None:
                rng  = np.random.default_rng(123)
                sim  = rng.dirichlet(a, size=min(1500, max(300, int(len(X)*1.5))))
                xyzs = _bary_to_xyz16(sim)
                n_sim = int(xyzs.shape[0])
                ax.scatter(xyzs[:,0], xyzs[:,1], xyzs[:,2], s=8, alpha=0.16, marker='o', label=f"Dirichlet sim (n={n_sim})")
    
        idx = {"z1":0, "z2":1, "z3":2}[color_by]
        sc  = ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], s=12, alpha=0.85, marker='o', c=Z[:,idx])
        cb  = plt.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(f"{color_by} (pivot-ILR)")
    
        ax.set_title(f"{block}: {sheet} — fargekodet {color_by}")
        if add_dirichlet_faint:
            ax.legend(loc='best', fontsize=8, framealpha=0.6)
    
        out = Path(out_dir) / f"{_safe_name(Path(xlsx_path).stem)}__{_safe_name(block)}__{_safe_name(sheet)}__color_by_{color_by}.png"
        plt.tight_layout(); plt.savefig(out, dpi=180, bbox_inches='tight'); plt.close(fig)
        print(f"[STEG 16C] Lagret: {out}")
        register_output(step="STEG 16C", label="result_fig", path=out, kind="png")
    
    # ---------- Kjøring av (A), (B) og (C) ----------
    _out_dir16 = Path(xlsx_path).with_name(Path(xlsx_path).stem + "_tetra_ilr")
    _out_dir16.mkdir(parents=True, exist_ok=True)
    
    # (A) Metode-demo (syntetisk). Bruk av OCAI-navn for lesbarhet.
    _methods_tetra_demo16(
        out_dir=_out_dir16,
        labels=["Klan","Adhockrati","Marked","Hierarki"],
        N=1000,
        alpha=(40,20,20,20),
        color_by="z1"
    )
    
    # (B) Resultat på virkelige data (krever DIR_RES + hjelpetjenester fra STEG 8)
    if not DIR_RES.empty:
        _results_tetra16(
            block="OCAI",
            sheet="MERGET_OCAI",
            labels=ocai_cols,
            out_dir=_out_dir16,
            title_extra="Pivot-ILR-akser"
        )
        _results_tetra16(
            block="Strategi",
            sheet="MERGET_STRATEGI",
            labels=strat_cols,
            out_dir=_out_dir16,
            title_extra="Pivot-ILR-akser"
        )
    else:
        print("[STEG 16B] Hoppet over resultatsfigurer: DIR_RES mangler/er tom.")
    
    # (C) Real-data fargelagt etter z1/z2/z3 (ingen Dirichlet-overlay som default)
    for block, sheet, labels in [
        ("OCAI",      "MERGET_OCAI",      ocai_cols),
        ("Strategi",  "MERGET_STRATEGI",  strat_cols),
    ]:
        for zlab in ("z1", "z2", "z3"):
            _results_tetra_color_by_z16(block, sheet, labels, _out_dir16, color_by=zlab, add_dirichlet_faint=False)

    if _DIRICHLET_SIM_META:
        try:
            with pd.ExcelWriter(_out_dir, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
                export_excel(pd.DataFrame(_DIRICHLET_SIM_META), writer=w, sheet_name="RUN_META", label="RUN_META")
        except Exception:
            pass

    # Går gjennom forventede utdatafiler/-mapper fra tidligere steg og oppsummerer hva som faktisk finnes.
    # Input er base_path/xlsx_path, output er en artefakt-CSV og -TXT samt konsolloversikt.
    # ============================
    # STEG 17: Artefakt-rapport (filer og mapper produsert)
    # ============================
    # Formål: Gi en rask oversikt over hva som ble skrevet ut av tidligere steg,
    # uten å endre tidligere funksjonalitet. Rapporter lagres som:
    #   <eksempeldatasett>_artefakt_rapport.csv
    #   <eksempeldatasett>_artefakt_rapport.txt
    #
    # Avhenger kun av xlsx_path/base_path fra STEG 1.
    
    # Rot og fil-stamme
    _root17   = Path(xlsx_path).parent
    _stem17   = Path(xlsx_path).stem
    
    # Mønstre for typiske artefakter fra forrige steg
    _patterns17 = [
        f"{_stem17}_p+bg.xlsx",
        f"{_stem17}_summary_overall.xlsx",
        f"{_stem17}_icc.xlsx",
        f"{_stem17}_icc.csv",
        f"{_stem17}_by_category_summary.xlsx",
        f"{_stem17}_dirichlet_alpha.xlsx",
        f"{_stem17}_sensitivitet_og_korrelasjoner.xlsx",
        f"{_stem17}_ilr_regresjoner.xlsx",
        f"{_stem17}_ilr_beta_tests.xlsx",
        f"{_stem17}_ilr_spearman_manova.xlsx",
        f"{_stem17}_reliability_across_sections_ILR.xlsx",
        f"{_stem17}_spearman_ILR_sensitivity.xlsx",
        # MANOVA-varianter
        f"{_stem17}_MANOVA_OCAI_byDept.xlsx",
        f"{_stem17}_MANOVA_STRAT_byDept.xlsx",
        f"{_stem17}_MANOVA_OCAI_byRole.xlsx",
        f"{_stem17}_MANOVA_STRAT_byRole.xlsx",
    ]
    
    # Kataloger som kan inneholde bilder
    _dirs17 = [
        _root17 / f"{_stem17}_dirichlet_plots",
        _root17 / f"{_stem17}_tetra_ilr",
    ]
    
    def _fmt_dt17(ts: float) -> str:
        try:
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""
    
    rows17 = []
    
    # 1) Samle filer som matcher mønstre
    for pat in _patterns17:
        p = _root17 / pat
        if p.exists() and p.is_file():
            stat = p.stat()
            rows17.append({
                "type": "file",
                "name": p.name,
                "path": str(p),
                "size_bytes": stat.st_size,
                "modified": _fmt_dt17(stat.st_mtime),
            })
    
    # 2) Samle kataloger og innhold (kun toppnivå)
    for d in _dirs17:
        if d.exists() and d.is_dir():
            dstat = d.stat()
            rows17.append({
                "type": "dir",
                "name": d.name,
                "path": str(d),
                "size_bytes": "",
                "modified": _fmt_dt17(dstat.st_mtime),
            })
            # List filer i mappen (ikke rekursivt)
            for child in sorted(d.glob("*")):
                if child.is_file():
                    cstat = child.stat()
                    rows17.append({
                        "type": "dir_file",
                        "name": child.name,
                        "path": str(child),
                        "size_bytes": cstat.st_size,
                        "modified": _fmt_dt17(cstat.st_mtime),
                    })
    
    # 3) Skriv rapport (CSV + TXT) dersom noe finnes
    _report_csv17 = _root17 / f"{_stem17}_artefakt_rapport.csv"
    _report_txt17 = _root17 / f"{_stem17}_artefakt_rapport.txt"
    
    if rows17:
        df17 = pd.DataFrame(rows17, columns=["type","name","path","size_bytes","modified"])
        df17.to_csv(_report_csv17, index=False, encoding="utf-8")
        register_output(step="STEG 17", label="artefakt_rapport_csv", path=_report_csv17, kind="csv", df=df17)
        # Liten tekstlig oppsummering
        lines = []
        lines.append(f"# Artefakt-rapport for '{_stem17}'")
        lines.append(f"Laget: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        for r in rows17:
            size = r["size_bytes"]
            size_str = f"{size} B" if isinstance(size, int) and size >= 0 else ""
            lines.append(f"{r['type']:9} | {r['modified']:19} | {size_str:>10} | {r['path']}")
        _report_txt17.write_text("\n".join(lines), encoding="utf-8")
        register_output(step="STEG 17", label="artefakt_rapport_txt", path=_report_txt17, kind="txt")
    
        print(f"[STEG 17] Skrev artefakt-rapport:")
        print(f" - CSV: {_report_csv17}")
        print(f" - TXT: {_report_txt17}")
    else:
        print("[STEG 17] Fant ingen artefakter å rapportere (ingen av forventede filer/mapper eksisterer).")
    
    # Sammenligner ilr-formelen direkte mot Ψ-matrisen for å sjekke numerisk samsvar på syntetiske data.
    # Input er genererte P/Ψ fra skriptet, output er en sanity-tabell skrevet til Excel + konsollvarsling.
    # ===========================
    # STEG 18: ILR-sanity check (FORMEL vs. Ψ-matrise)
    # ===========================
    # - For hvert ark (OCAI/Strategi) og for MERGET_*:
    #   * bygger P (0..1, sum≈1, >0)
    #   * beregner z via lukkede formler (pivot-basis)
    #   * beregner z via clr(P) @ Ψ (samme pivot-basis)
    #   * sammenligner |Δ| og rapporterer maks-avvik per koordinat
    # - Skriver resultat til <eksempeldatasett>_ilr_sanity.xlsx
    
    # Pivot-Ψ for rekkefølge (p1,p2,p3,p4) — samme som brukt ellers i skriptet
    _PSI_SANITY = np.array([
        [ np.sqrt(3/4),         0.0,                 0.0               ],  # z1: p1 vs {p2,p3,p4}
        [-1/np.sqrt(12),  np.sqrt(2/3),              0.0               ],  # z2: p2 vs {p3,p4}
        [-1/np.sqrt(12), -1/np.sqrt(6),        1/np.sqrt(2)           ],  # z3: p3 vs p4
        [-1/np.sqrt(12), -1/np.sqrt(6),       -1/np.sqrt(2)           ],
    ])
    
    # Sanity: Ψ must be orthonormal (catches accidental edits)
    assert np.allclose(_PSI_SANITY.T @ _PSI_SANITY, np.eye(3)), "Ψ is not orthonormal"
    
    def _clean_P_matrix(df: pd.DataFrame, cols, tol=1e-6, eps=1e-12):
        """
        Plukker ut kolonnene 'cols', skalerer til 0..1 ved behov, beholder rader med sum≈1,
        erstatter ikke-positive med eps og re-normaliserer.
        Returnerer ndarray (n,4).
        """
        if not set(cols).issubset(df.columns):
            return np.empty((0, len(cols)))
    
        X = df[cols].astype(float).to_numpy(copy=True)
        if X.size == 0:
            return X
    
        # Skaler 0..100 -> 0..1
        if np.nanmax(X) > 1.0:
            X = X / 100.0
    
        # Behold kun rader med sum≈1
        s = np.nansum(X, axis=1)
        keep = np.isfinite(s) & (np.abs(s - 1.0) <= tol)
        X = X[keep]
        if X.size == 0:
            return X
    
        # Erstatt NaN/<=0, vakt mot nullrader, og re-normaliser
        X[np.isnan(X)] = 0.0
        X[X <= 0.0] = eps
        row_sums = X.sum(axis=1, keepdims=True)
        keep2 = np.isfinite(row_sums).ravel() & (row_sums.ravel() > 0.0)
        X = X[keep2]
        if X.size == 0:
            return X
        X = X / X.sum(axis=1, keepdims=True)
        return X
    
    def _ilr_pivot_formula(P: np.ndarray):
        """
        Lukkede formler for pivot-ILR (p1,p2,p3,p4):
          z1 = √(3/4)*(ln p1 - (ln p2+ln p3+ln p4)/3)
          z2 = √(2/3)*(ln p2 - (ln p3+ln p4)/2)
          z3 = √(1/2)*(ln p3 - ln p4)
        """
        P = np.asarray(P, float)
        if P.size == 0:
            return P.reshape(0, 3)
        L1, L2, L3, L4 = np.log(P[:,0]), np.log(P[:,1]), np.log(P[:,2]), np.log(P[:,3])
        z1 = np.sqrt(3/4.0) * (L1 - (L2 + L3 + L4)/3.0)
        z2 = np.sqrt(2/3.0) * (L2 - (L3 + L4)/2.0)
        z3 = np.sqrt(1/2.0) * (L3 - L4)
        return np.column_stack([z1, z2, z3])
    
    def _ilr_pivot_matrix(P: np.ndarray, psi=_PSI_SANITY):
        """
        Matriseform: z = clr(P) @ Ψ, med samme pivot-Ψ som over.
        """
        P = np.asarray(P, float)
        if P.size == 0:
            return P.reshape(0, 3)
        L = np.log(P)
        clr = L - L.mean(axis=1, keepdims=True)
        return clr @ psi
    
    def _merged_P_for_block(block_name: str) -> pd.DataFrame:
        """
        Bygger ett MERGET_* P-dataframe for 'OCAI' eller 'Strategi' fra p_tables.
        """
        cols = ocai_cols if block_name == "OCAI" else strat_cols
        frames = []
        for sh, P in p_tables.items():
            if set(cols).issubset(P.columns):
                frames.append(P[cols].copy())
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, axis=0, ignore_index=True)
    
    def _sanity_for_df(df: pd.DataFrame, cols, label: str, tol_abs=1e-10):
        """
        Kjører sanity for ett datasett (ark eller MERGET). Returnerer én rad for rapport.
        """
        P = _clean_P_matrix(df, cols)
        if P.size == 0:
            return {
                "Blokk": "OCAI" if cols == ocai_cols else "Strategi",
                "Ark": label,
                "N_checked": 0,
                "max_abs_diff_z1": np.nan,
                "max_abs_diff_z2": np.nan,
                "max_abs_diff_z3": np.nan,
                "max_abs_diff_all": np.nan,
                "PASS(tol)": False,
                "tol_abs": tol_abs
            }
        Zf = _ilr_pivot_formula(P)
        Zm = _ilr_pivot_matrix(P, _PSI_SANITY)
        diffs = np.abs(Zf - Zm)
        mx = diffs.max(axis=0)
        mx_all = float(np.max(mx))
        return {
            "Blokk": "OCAI" if cols == ocai_cols else "Strategi",
            "Ark": label,
            "N_checked": int(P.shape[0]),
            "max_abs_diff_z1": float(mx[0]),
            "max_abs_diff_z2": float(mx[1]),
            "max_abs_diff_z3": float(mx[2]),
            "max_abs_diff_all": mx_all,
            "PASS(tol)": bool(mx_all <= tol_abs),
            "tol_abs": tol_abs
        }
    
    rows = []
    
    # Per-ark sjekk
    for sh, Pdf in p_tables.items():
        if set(ocai_cols).issubset(Pdf.columns):
            rows.append(_sanity_for_df(Pdf, ocai_cols, sh))
        if set(strat_cols).issubset(Pdf.columns):
            rows.append(_sanity_for_df(Pdf, strat_cols, sh))
    
    # MERGET_* sjekk (fra p_tables)
    _ocai_merge_df = _merged_P_for_block("OCAI")
    _strat_merge_df = _merged_P_for_block("Strategi")
    if not _ocai_merge_df.empty:
        rows.append(_sanity_for_df(_ocai_merge_df, ocai_cols, "MERGET_OCAI"))
    if not _strat_merge_df.empty:
        rows.append(_sanity_for_df(_strat_merge_df, strat_cols, "MERGET_STRATEGI"))
    
    ILR_SANITY = pd.DataFrame(rows, columns=[
        "Blokk","Ark","N_checked","max_abs_diff_z1","max_abs_diff_z2","max_abs_diff_z3",
        "max_abs_diff_all","PASS(tol)","tol_abs"
    ])
    
    print("\n[STEG 18] ILR sanity (FORMEL vs. Ψ): maks |Δ| per koordinat")
    if ILR_SANITY.empty:
        print("Ingen datasett å sjekke.")
    else:
        disp = ILR_SANITY.copy()
        for c in ["max_abs_diff_z1","max_abs_diff_z2","max_abs_diff_z3","max_abs_diff_all"]:
            disp[c] = pd.to_numeric(disp[c], errors="coerce").round(12)
        _log_df_meta("[STEG 18] ILR_SANITY (meta)", disp)
    
    # Lagre til Excel
    _out_sanity = base_path.with_name(base_path.stem + "_ilr_sanity.xlsx")
    try:
        with pd.ExcelWriter(_out_sanity, engine="xlsxwriter") as w:
            export_excel(ILR_SANITY, writer=w, sheet_name="ILR_sanity", label="ILR_sanity")
        print(f"[STEG 18] Lagret sanity-tabell til: {_out_sanity}")
        register_output(step="STEG 18", label="ilr_sanity", path=_out_sanity, kind="xlsx", df=ILR_SANITY)
    except Exception:
        with pd.ExcelWriter(_out_sanity, engine="openpyxl") as w:
            export_excel(ILR_SANITY, writer=w, sheet_name="ILR_sanity", label="ILR_sanity")
        print(f"[STEG 18] Lagret sanity-tabell til (openpyxl): {_out_sanity}")
        register_output(step="STEG 18", label="ilr_sanity", path=_out_sanity, kind="xlsx", df=ILR_SANITY, note="openpyxl")

    if RUN_MODE == "DEV":
        print(f"[HC3] HC3->HC1 fallbacks in DEV: {_robust_fallback_counter['count']}")

if __name__ == "__main__":
    run_all_steps()

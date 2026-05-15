"""
H-AIPW with IST as external data source.
ATE = θ̂₁ − θ̂₀, optimal λ estimated per arm.
"""

import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent.parent
IST3_PATH = BASE / "two paper" / "IST3 dataset" / "datashare_aug2015.sas7bdat"
IST_PATH  = BASE / "two paper" / "IST dataset"  / "IST_corrected.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
N_SPLITS      = 30      # cross-fitting folds (following De Bartolomeis et al.)
N_REPS        = 1000
SAMPLE_SIZES  = [100, 200, 344, 688, 1376]   # halving sequence from full n=2752
SEED          = 42
REF_IST3_DM   = None    # set at runtime by run_baseline()
Z95           = 1.96

# All 18 IST3 covariates for INTERNAL outcome model
IST3_COVARIATES = [
    "age", "gender", "randdelay", "sbprand", "nihss",
    "gcs_score_rand", "atrialfib_rand", "stroketype", "weight", "glucose",
    "weakface_rand", "weakarm_rand", "weakleg_rand", "dysphasia_rand",
    "hemianopia_rand", "visuospat_rand", "brainstemsigns_rand", "otherdeficit_rand",
]

# 14 harmonised covariates shared with IST (used for external models)
SHARED_FEATURES = [
    "age", "sex", "randdelay", "sbprand",
    "weakface", "weakarm", "weakleg", "dysphasia",
    "hemianopia", "visuospat", "brainstem", "otherdeficit",
    "atrial", "stroketype",
]

LR_KWARGS = dict(max_iter=2000, solver="lbfgs", random_state=SEED)

def make_lr():
    """Logistic regression with standard scaling (helps convergence on mixed-scale covariates)."""
    return make_pipeline(StandardScaler(), LogisticRegression(**LR_KWARGS))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ist3():
    print("Loading IST3...")
    df, _ = pyreadstat.read_sas7bdat(str(IST3_PATH))
    required = IST3_COVARIATES + ["itt_treat", "ohs6"]
    df = df.dropna(subset=required).reset_index(drop=True)
    df["randyear"]  = df["randyear"].astype(float)
    df["randmonth"] = df["randmonth"].astype(float)
    df["Y"] = (df["ohs6"] <= 2).astype(float)   # alive and independent (OHS<=2)
    df["A"] = df["itt_treat"].astype(float)

    binary_map = {
        "gender":              "sex",
        "atrialfib_rand":      "atrial",
        "weakface_rand":       "weakface",
        "weakarm_rand":        "weakarm",
        "weakleg_rand":        "weakleg",
        "dysphasia_rand":      "dysphasia",
        "hemianopia_rand":     "hemianopia",
        "visuospat_rand":      "visuospat",
        "brainstemsigns_rand": "brainstem",
        "otherdeficit_rand":   "otherdeficit",
    }
    for src, dst in binary_map.items():
        df[dst] = (df[src] == 1).astype(float)
    df["randdelay"]  = df["randdelay"].astype(float)
    df["sbprand"]    = df["sbprand"].astype(float)
    df["age"]        = df["age"].astype(float)
    df["stroketype"] = df["stroketype"].astype(float)

    print(f"  IST3 n={len(df)}, outcome={df.Y.mean():.3f}, "
          f"treated={(df.A==1).sum()}, control={(df.A==0).sum()}")
    return df


def load_ist():
    print("Loading IST...")
    df = pd.read_csv(str(IST_PATH), encoding="latin1", low_memory=False)
    df = df[df["OCCODE"].isin([1, 2, 3, 4])].copy()
    # OCCODE=2 (dependent) and =4 (recovered) → Y=1; broader than IST3 OHS≤2 but H-AIPW down-weights miscalibrated models
    df["Y"] = (df["OCCODE"].isin([2, 4])).astype(float)

    needed = ["AGE", "SEX", "RDELAY", "RSBP",
              "RDEF1", "RDEF2", "RDEF3", "RDEF4",
              "RDEF5", "RDEF6", "RDEF7", "RDEF8",
              "RATRIAL", "STYPE", "RXASP", "RXHEP", "Y"]
    df = df.dropna(subset=needed).reset_index(drop=True)

    df["A_ist"]     = (~((df["RXASP"] == "N") & (df["RXHEP"] == "N"))).astype(float)
    df["A_asp_only"] = ((df["RXASP"] == "Y") & (df["RXHEP"] == "N")).astype(float)

    df["age"]       = df["AGE"].astype(float)
    df["sex"]       = (df["SEX"] == "M").astype(float)
    df["randdelay"] = df["RDELAY"].astype(float)
    df["sbprand"]   = df["RSBP"].astype(float)
    df["atrial"]    = (df["RATRIAL"] == "Y").astype(float)
    for i, col in enumerate(["weakface", "weakarm", "weakleg", "dysphasia",
                              "hemianopia", "visuospat", "brainstem", "otherdeficit"], 1):
        df[col] = (df[f"RDEF{i}"] == "Y").astype(float)
    stype_map = {"TACS": 1, "PACS": 2, "POCS": 3, "LACS": 4, "OTH": 5}
    df["stroketype"] = df["STYPE"].map(stype_map).fillna(5).astype(float)

    ctrl     = df[df["A_ist"] == 0]
    trt      = df[df["A_ist"] == 1]
    asp_only = df[df["A_asp_only"] == 1]
    print(f"  IST n={len(df)}, ctrl={len(ctrl)} (rate={ctrl.Y.mean():.3f}), "
          f"trt={len(trt)} (rate={trt.Y.mean():.3f}), "
          f"asp-only={len(asp_only)} (rate={asp_only.Y.mean():.3f})")
    return df


# ── Arm-specific influence functions ──────────────────────────────────────────

def phi1(Y, A, Q1, pi):
    """Treatment-arm influence function: estimates E[Y(1)]."""
    return (A / pi) * (Y - Q1) + Q1

def phi0(Y, A, Q0, pi):
    """Control-arm influence function: estimates E[Y(0)]."""
    return ((1 - A) / (1 - pi)) * (Y - Q0) + Q0


# ── Internal cross-fitted outcome models (logistic regression) ─────────────────

def crossfit_arm_models(Y, A, X, n_splits=N_SPLITS):
    """Cross-fitted logistic regression. Returns Q1_hat, Q0_hat — P(Y=1|X,a)."""
    n = len(Y)
    Q1_hat = np.zeros(n)
    Q0_hat = np.zeros(n)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    for tr, te in kf.split(X):
        X_tr, Y_tr, A_tr = X[tr], Y[tr], A[tr]
        X_te = X[te]

        mask1 = A_tr == 1
        if mask1.sum() >= 2:
            m1 = make_lr().fit(X_tr[mask1], Y_tr[mask1])
            Q1_hat[te] = m1.predict_proba(X_te)[:, 1]

        mask0 = A_tr == 0
        if mask0.sum() >= 2:
            m0 = make_lr().fit(X_tr[mask0], Y_tr[mask0])
            Q0_hat[te] = m0.predict_proba(X_te)[:, 1]

    return Q1_hat, Q0_hat


# ── Arm-specific H-AIPW combination ───────────────────────────────────────────

def haipw_arm(phi_list):
    """Optimal combination of arm-specific IFs. Returns (theta_a, scaled_var_a, lam_a)."""
    Psi   = np.column_stack(phi_list)
    Sigma = np.cov(Psi.T)

    if Psi.shape[1] == 1:
        theta     = Psi.mean()
        scaled_v  = float(np.var(Psi[:, 0], ddof=1))
        return theta, scaled_v, np.array([1.0])

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma    += 1e-6 * np.eye(Sigma.shape[0])
        Sigma_inv = np.linalg.inv(Sigma)

    ones = np.ones(Psi.shape[1])
    lam  = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

    theta      = float(lam @ Psi.mean(axis=0))
    scaled_var = float(lam @ Sigma @ lam)

    return theta, scaled_var, lam


def haipw_ate(phi1_list, phi0_list):
    """H-AIPW ATE = θ̂₁ − θ̂₀. Returns (ATE, scaled_var, lam1, lam0)."""
    theta1, sv1, lam1 = haipw_arm(phi1_list)
    theta0, sv0, lam0 = haipw_arm(phi0_list)

    Psi1  = np.column_stack(phi1_list) @ lam1
    Psi0  = np.column_stack(phi0_list) @ lam0
    cov01 = float(np.cov(Psi1, Psi0)[0, 1])

    return theta1 - theta0, sv1 + sv0 - 2 * cov01, lam1, lam0


# ── External model training ────────────────────────────────────────────────────

def train_external_models(ist_df):
    """Train logistic regression on IST aspirin-only arm (RXASP=Y, RXHEP=N)."""
    X = ist_df[SHARED_FEATURES].values
    Y = ist_df["Y"].values
    Asp = ist_df["A_asp_only"].values

    m_asp = make_pipeline(StandardScaler(), LogisticRegression(**LR_KWARGS)).fit(X[Asp == 1], Y[Asp == 1])

    print(f"  External model trained: IST aspirin-only n={(Asp==1).sum()} (rate={Y[Asp==1].mean():.3f})")
    return m_asp


# ── Main Analysis: Random Subsampling ──────────────────────────────────────────

def run_subsampling(ist3_df, m_asp, ref_dm=None):
    """Random subsampling simulation. ref_dm is the full-sample DM reference for MSE/coverage."""
    if ref_dm is None:
        raise ValueError("ref_dm must be provided — run run_baseline() first.")
    rng = np.random.default_rng(SEED)

    Y_all = ist3_df["Y"].values
    A_all = ist3_df["A"].values
    X_int = ist3_df[IST3_COVARIATES].values
    X_ext = ist3_df[SHARED_FEATURES].values

    treat_idx   = np.where(A_all == 1)[0]
    control_idx = np.where(A_all == 0)[0]

    results = []

    for n in SAMPLE_SIZES:
        n1 = n // 2; n0 = n - n1
        print(f"  n={n} ({N_REPS} reps)...", flush=True)

        sv_dm, sv_aipw, sv_haipw = [], [], []
        ate_dm, ate_aipw, ate_haipw = [], [], []
        lam1_all, lam0_all = [], []
        cov_dm, cov_aipw, cov_haipw = [], [], []

        for rep in range(N_REPS):
            idx1 = rng.choice(treat_idx,   n1, replace=False)
            idx0 = rng.choice(control_idx, n0, replace=False)
            idx  = np.concatenate([idx1, idx0])

            Yi, Ai = Y_all[idx], A_all[idx]
            Xi_int = X_int[idx]
            Xi_ext = X_ext[idx]
            pi = 0.5   # known by design (balanced randomization in IST3)

            # DM
            dm   = Yi[Ai == 1].mean() - Yi[Ai == 0].mean()
            v_dm = (np.var(Yi[Ai==1], ddof=1) / (Ai==1).sum() +
                    np.var(Yi[Ai==0], ddof=1) / (Ai==0).sum())

            # AIPW
            Q1_int, Q0_int = crossfit_arm_models(Yi, Ai, Xi_int)
            psi1_int = phi1(Yi, Ai, Q1_int, pi)
            psi0_int = phi0(Yi, Ai, Q0_int, pi)
            theta_aipw = psi1_int.mean() - psi0_int.mean()
            if_cov = np.cov(psi1_int, psi0_int)[0, 1]
            v_aipw = (np.var(psi1_int, ddof=1) + np.var(psi0_int, ddof=1) - 2 * if_cov)

            # H-AIPW with IST external model
            Q_asp    = m_asp.predict_proba(Xi_ext)[:, 1]
            psi1_asp = phi1(Yi, Ai, Q_asp, pi)
            psi0_asp = phi0(Yi, Ai, Q_asp, pi)
            ate_h, sv_h, l1_h, l0_h = haipw_ate(
                [psi1_int, psi1_asp], [psi0_int, psi0_asp])
            lam1_all.append(l1_h); lam0_all.append(l0_h)

            # Coverage
            se_dm    = np.sqrt(v_dm)
            se_aipw  = np.sqrt(v_aipw / n)
            se_haipw = np.sqrt(sv_h   / n)
            cov_dm.append(int(abs(dm           - ref_dm) <= Z95 * se_dm))
            cov_aipw.append(int(abs(theta_aipw - ref_dm) <= Z95 * se_aipw))
            cov_haipw.append(int(abs(ate_h     - ref_dm) <= Z95 * se_haipw))

            sv_dm.append(n * v_dm)
            sv_aipw.append(v_aipw)
            sv_haipw.append(sv_h)
            ate_dm.append(dm)
            ate_aipw.append(theta_aipw)
            ate_haipw.append(ate_h)

        def mse(ates):
            return np.mean((np.array(ates) - ref_dm) ** 2)

        l1 = np.mean([l[1] for l in lam1_all])
        l0 = np.mean([l[1] for l in lam0_all])

        results.append({
            "n":             n,
            "sv_DM":         np.mean(sv_dm),
            "sv_AIPW":       np.mean(sv_aipw),
            "sv_HAIPW":      np.mean(sv_haipw),
            "mse_DM":        mse(ate_dm),
            "mse_AIPW":      mse(ate_aipw),
            "mse_HAIPW":     mse(ate_haipw),
            "ate_DM":        np.mean(ate_dm),
            "ate_AIPW":      np.mean(ate_aipw),
            "lam1_ext":      round(l1, 3),
            "lam0_ext":      round(l0, 3),
            "cov_DM":        round(np.mean(cov_dm),    3),
            "cov_AIPW":      round(np.mean(cov_aipw),  3),
            "cov_HAIPW":     round(np.mean(cov_haipw), 3),
        })

        print(f"    DM={np.mean(sv_dm):.3f}  AIPW={np.mean(sv_aipw):.3f}  "
              f"H-AIPW={np.mean(sv_haipw):.3f}  "
              f"λ₁={l1:.3f}  λ₀={l0:.3f}")

        dist_rows = [{"n": n, "rep": r,
                      "ate_DM": ate_dm[r], "ate_AIPW": ate_aipw[r],
                      "ate_HAIPW": ate_haipw[r]}
                     for r in range(N_REPS)]
        dist_path = Path(__file__).parent / "ate_distributions.csv"
        header = not dist_path.exists() or n == SAMPLE_SIZES[0]
        pd.DataFrame(dist_rows).to_csv(
            dist_path, mode='w' if header else 'a', header=header, index=False)

    return pd.DataFrame(results)


# ── Print results ──────────────────────────────────────────────────────────────

def print_results(sub_df, ref_dm=None):
    print("\n" + "="*80)
    print("TABLE: Scaled Asymptotic Variance (n·Var[θ̂]) — Arm-Specific H-AIPW")
    print("="*80)
    print(f"{'Estimator':<22}", end="")
    for _, r in sub_df.iterrows():
        print(f"  {'n='+str(int(r.n)):>12}", end="")
    print()
    print("-"*80)
    for label, key in [
        ("DM",              "sv_DM"),
        ("AIPW",      "sv_AIPW"),
        ("H-AIPW",    "sv_HAIPW"),
    ]:
        print(f"{label:<22}", end="")
        for _, r in sub_df.iterrows():
            print(f"  {r[key]:>12.3f}", end="")
        print()

    print("\n" + "="*60)
    print("TABLE: Variance Reduction vs AIPW (%)")
    print("="*60)
    print(f"{'H-AIPW':<22}", end="")
    for _, r in sub_df.iterrows():
        pct = 100 * (r["sv_AIPW"] - r["sv_HAIPW"]) / r["sv_AIPW"]
        print(f"  {pct:>7.1f}%", end="")
    print()

    print("\n" + "="*60)
    print("TABLE: Mean Optimal Lambda (external weight, averaged over reps)")
    print("="*60)
    print(f"{'Arm':<30}", end="")
    for _, r in sub_df.iterrows():
        print(f"  {'n='+str(int(r.n)):>8}", end="")
    print()
    for label, key in [
        ("λ₁ (trt arm, IST aspirin)", "lam1_ext"),
        ("λ₀ (ctrl arm, IST aspirin)", "lam0_ext"),
    ]:
        print(f"{label:<30}", end="")
        for _, r in sub_df.iterrows():
            print(f"  {r[key]:>8.3f}", end="")
        print()

    print("\n" + "="*70)
    ref_label = f"{ref_dm:.4f}" if ref_dm is not None else "N/A"
    print(f"TABLE: Empirical Coverage Rate (nominal 95%, ref = IST3 DM {ref_label})")
    print("="*70)
    header = f"{'Estimator':<22}" + "".join(f"  {'n='+str(int(r.n)):>8}" for _, r in sub_df.iterrows())
    print(header)
    print("-"*70)
    for label, key in [("DM", "cov_DM"), ("AIPW", "cov_AIPW"), ("H-AIPW", "cov_HAIPW")]:
        print(f"  {label:<20}", end="")
        for _, r in sub_df.iterrows():
            print(f"  {r[key]*100:>7.1f}%", end="")
        print()


def run_baseline(ist3_df, m_asp):
    """Single estimate on full IST3 complete-case sample (n=2752)."""
    Y  = ist3_df["Y"].values
    A  = ist3_df["A"].values
    Xi = ist3_df[IST3_COVARIATES].values
    Xe = ist3_df[SHARED_FEATURES].values
    n  = len(Y)
    pi = 0.5   # known by design (balanced randomization in IST3)

    dm   = Y[A==1].mean() - Y[A==0].mean()
    v_dm = (np.var(Y[A==1], ddof=1)/(A==1).sum() +
            np.var(Y[A==0], ddof=1)/(A==0).sum())

    Q1_int, Q0_int = crossfit_arm_models(Y, A, Xi)
    psi1_int = phi1(Y, A, Q1_int, pi)
    psi0_int = phi0(Y, A, Q0_int, pi)
    theta_aipw = psi1_int.mean() - psi0_int.mean()
    cov_c  = np.cov(psi1_int, psi0_int)[0, 1]
    v_aipw = np.var(psi1_int, ddof=1) + np.var(psi0_int, ddof=1) - 2*cov_c

    Q_asp    = m_asp.predict_proba(Xe)[:, 1]
    psi1_asp = phi1(Y, A, Q_asp, pi)
    psi0_asp = phi0(Y, A, Q_asp, pi)
    ate_h, sv_h, lam1, lam0 = haipw_ate([psi1_int, psi1_asp], [psi0_int, psi0_asp])

    print(f"\n── BASELINE: FULL IST3 SAMPLE (n={n}) ──")
    print(f"  DM      ATE={dm:.4f}  SE={np.sqrt(v_dm):.4f}")
    print(f"  AIPW    ATE={theta_aipw:.4f}  SE={np.sqrt(v_aipw/n):.4f}")
    print(f"  H-AIPW  ATE={ate_h:.4f}  SE={np.sqrt(sv_h/n):.4f}  λ₁={lam1[1]:.3f}  λ₀={lam0[1]:.3f}")
    print(f"  → Using full-sample DM as coverage reference: {dm:.4f}")
    return dm   # used as REF_IST3_DM in subsampling coverage / MSE


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ist3_df = load_ist3()
    ist_df  = load_ist()

    print("\nTraining external models on IST...")
    m_asp = train_external_models(ist_df)

    ref_dm = run_baseline(ist3_df, m_asp)

    print("\n── MAIN ANALYSIS: RANDOM SUBSAMPLING ──")
    sub_df = run_subsampling(ist3_df, m_asp, ref_dm=ref_dm)

    print_results(sub_df, ref_dm=ref_dm)

    out1 = Path(__file__).parent / "subsampling_results.csv"
    out2 = Path(__file__).parent / "ate_distributions.csv"
    sub_df.to_csv(out1, index=False)
    print(f"\nSaved:\n  {out1}\n  {out2}")

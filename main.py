"""
Run both H-AIPW analyses: IST external model and LLM external model.
Results are saved to subsampling_results.csv and haipw_llm_results.csv.
"""

from pathlib import Path

from haipw_analysis import (
    load_ist3, load_ist, train_external_models,
    run_baseline, run_subsampling, print_results,
)
from haipw_llm_analysis import (
    load_ist3 as load_ist3_llm,
    load_llm_predictions, run_subsampling as run_llm_subsampling,
)

OUT_DIR = Path(__file__).parent


def run_ist_analysis():
    """H-AIPW with IST (aspirin-only arm) as external model."""
    print("\n" + "="*60)
    print("IST EXTERNAL MODEL ANALYSIS")
    print("="*60)
    ist3_df = load_ist3()
    ist_df  = load_ist()
    m_asp   = train_external_models(ist_df)
    ref_dm  = run_baseline(ist3_df, m_asp)

    print("\n── SUBSAMPLING ──")
    sub_df = run_subsampling(ist3_df, m_asp, ref_dm=ref_dm)
    print_results(sub_df, ref_dm=ref_dm)

    sub_df.to_csv(OUT_DIR / "subsampling_results.csv", index=False)
    print(f"\nSaved: subsampling_results.csv")


def run_llm_analysis():
    """H-AIPW with LLM predictions as external model."""
    print("\n" + "="*60)
    print("LLM EXTERNAL MODEL ANALYSIS")
    print("="*60)
    df = load_ist3_llm()
    df = load_llm_predictions(df)

    Y, A = df["outcome"].values, df["treat"].values
    ref_dm = float(Y[A == 1].mean() - Y[A == 0].mean())
    print(f"IST3 DM reference: {ref_dm:.4f}")

    results = run_llm_subsampling(df, ref_dm=ref_dm)
    results.to_csv(OUT_DIR / "haipw_llm_results.csv", index=False)
    print(f"\nSaved: haipw_llm_results.csv")
    print(results[["n", "sv_DM", "sv_AIPW", "sv_HAIPW_LLM", "lam1_LLM", "lam0_LLM"]].to_string(index=False))


if __name__ == "__main__":
    run_ist_analysis()
    run_llm_analysis()

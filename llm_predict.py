"""
Query Claude/Gemini to predict IST3 stroke outcomes (LLM external model).
Produces f(X_i, 1) and f(X_i, 0) for each patient i.
Output: llm_predictions/predictions_<model>.csv
"""

import os
import json
import time
import argparse
import pandas as pd
import numpy as np
import pyreadstat
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent.parent / "two paper" / "IST3 dataset" / "datashare_aug2015.sas7bdat"
OUTPUT_DIR = Path(__file__).parent / "llm_predictions"
N_PROMPTS   = 5      # prompt variations per patient per condition
SLEEP_BETWEEN_CALLS = 1.0   # seconds between API calls

# ── Prompt templates ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert neurologist specialising in acute ischemic stroke outcomes. "
    "Your answer must be a single integer between 0 and 10, provided in JSON format "
    "with exactly one key called \"response\". Do not include any explanation."
)

# Pool of varied closing instructions (following De Bartolomeis et al. 2025)
INSTRUCTION_POOL = [
    "Based on your clinical expertise, rate the likelihood on the scale.",
    "Consider the clinical profile carefully and assign a score on the scale.",
    "Use your medical knowledge to evaluate the prognosis and choose a number.",
    "Reflect on all clinical factors and provide your rating on the scale.",
    "Given the patient's characteristics and treatment, quantify the likely outcome.",
]

TREATMENT_LABEL = {
    1: "intravenous rt-PA (alteplase, thrombolysis)",
    0: "standard supportive care without thrombolysis (control)",
}

STROKE_TYPE_LABEL = {
    1: "total anterior circulation stroke",
    2: "partial anterior circulation stroke",
    3: "posterior circulation stroke",
    4: "lacunar stroke",
    5: "stroke of undetermined type",
}


def build_user_prompt(row: pd.Series, condition: int, instruction: str) -> str:
    """Build patient prompt. condition: 1 = rt-PA, 0 = control."""
    sex      = "male" if row["gender"] == 1 else "female"
    af       = "yes" if row["atrialfib_rand"] == 1 else "no"
    stype    = STROKE_TYPE_LABEL.get(int(row["stroketype"]), "unknown type")
    treat    = TREATMENT_LABEL[condition]

    def deficit(val):  # 1=present, else absent
        return "present" if val == 1 else "absent"

    prompt = f"""A {int(row['age'])}-year-old {sex} patient has been admitted with acute ischemic stroke ({stype}).

Clinical profile at randomisation:
- Time from symptom onset to randomisation: {row['randdelay']:.1f} hours
- Systolic blood pressure: {int(row['sbprand'])} mmHg
- NIHSS score: {int(row['nihss'])} (higher = more severe)
- GCS total score: {int(row['gcs_score_rand'])} (out of 15)
- Atrial fibrillation: {af}
- Facial weakness: {deficit(row['weakface_rand'])}
- Arm/hand weakness: {deficit(row['weakarm_rand'])}
- Leg/foot weakness: {deficit(row['weakleg_rand'])}
- Dysphasia: {deficit(row['dysphasia_rand'])}
- Homonymous hemianopia: {deficit(row['hemianopia_rand'])}
- Visuospatial disorder: {deficit(row['visuospat_rand'])}
- Brainstem/cerebellar signs: {deficit(row['brainstemsigns_rand'])}

Treatment received: {treat}.

Outcome question: How likely is it that this patient will be alive and independent \
(Oxford Handicap Scale score 0–2, meaning able to look after own affairs without assistance) \
at 6 months after stroke onset?

Choose an integer between 0 (very unlikely) and 10 (very likely).
{instruction}"""
    return prompt


# ── Data loading ───────────────────────────────────────────────────────────────

def load_ist3_complete_cases() -> pd.DataFrame:
    """Load IST3 complete-case population."""
    print("Loading IST3 data...")
    df, _ = pyreadstat.read_sas7bdat(str(DATA_PATH))

    required_vars = [
        "itt_treat", "age", "gender", "randdelay", "sbprand", "nihss",
        "gcs_score_rand", "atrialfib_rand", "stroketype", "weight", "glucose",
        "weakface_rand", "weakarm_rand", "weakleg_rand", "dysphasia_rand",
        "hemianopia_rand", "visuospat_rand", "brainstemsigns_rand",
        "otherdeficit_rand", "ohs6",
    ]
    df = df.dropna(subset=required_vars).reset_index(drop=True)
    df["pat_id"] = df.index
    df["outcome"] = (df["ohs6"] <= 5).astype(int)   # OHS<=5: alive at 6m

    print(f"Complete-case population: n={len(df)}")
    print(f"  rt-PA: {(df.itt_treat==1).sum()}, Control: {(df.itt_treat==0).sum()}")
    print(f"  Alive at 6m (OHS<=5): {df.outcome.sum()} ({df.outcome.mean():.1%})")
    return df


# ── API callers ────────────────────────────────────────────────────────────────

def call_claude(system: str, user: str, api_key: str) -> int | None:
    """Call Claude API. Returns integer 0-10, or None on failure."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=50,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = message.content[0].text.strip()
        try:
            val = int(json.loads(text)["response"])
        except (json.JSONDecodeError, KeyError):
            import re
            nums = re.findall(r'\b(\d+)\b', text)
            if not nums:
                print(f"    [Claude] unexpected response: {repr(text)}")
                return None
            val = int(nums[0])
        return max(0, min(10, val))
    except Exception as e:
        err = str(e)
        if "529" in err or "overloaded" in err.lower():
            print(f"    [Claude overloaded] waiting 30s before retry...")
            time.sleep(30)
            return call_claude(system, user, api_key)
        print(f"    [Claude error] {e}")
        return None


def call_gemini(system: str, user: str, api_key: str) -> int | None:
    """Call Gemini API. Returns integer 0-10, or None on failure. Exponential backoff on 429."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    max_retries = 5
    wait = 15   # free tier: ~15 req/min
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    max_output_tokens=20,
                ),
                contents=user,
            )
            text = response.text.strip()
            parsed = json.loads(text)
            val = int(parsed["response"])
            return max(0, min(10, val))
        except Exception as e:
            err = str(e)
            if "429" in err or "TooManyRequests" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
                print(f"    [Rate limit] waiting {wait}s before retry {attempt+1}/{max_retries}...")
                time.sleep(wait)
                wait *= 2
            else:
                print(f"    [Gemini error] {e}")
                return None
    print("    [Gemini] max retries exceeded, skipping this call.")
    return None


# ── Main prediction loop ───────────────────────────────────────────────────────

def predict(df: pd.DataFrame, model_name: str, api_key: str, n_patients: int) -> pd.DataFrame:
    """Run LLM predictions for n_patients patients × 2 conditions. Resumes from partial output."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"predictions_{model_name}.csv"

    done_ids = set()
    if out_path.exists():
        prev = pd.read_csv(out_path)
        prev_valid = prev.dropna(subset=["predicted_prob"])
        done_ids = set(zip(prev_valid["pat_id"], prev_valid["condition"]))
        print(f"Resuming: {len(prev_valid)} valid rows already saved, {len(done_ids)//2} patients done.")

    caller = call_claude if model_name == "claude" else call_gemini

    subset = df.head(n_patients) if isinstance(n_patients, int) else df

    rows = []
    total = len(subset) * 2
    done  = 0

    for _, patient in subset.iterrows():
        pid = int(patient["pat_id"])

        for cond in [0, 1]:
            if (pid, cond) in done_ids:
                done += 1
                continue

            responses = []
            for instr in INSTRUCTION_POOL[:N_PROMPTS]:
                user_prompt = build_user_prompt(patient, cond, instr)
                val = caller(SYSTEM_PROMPT, user_prompt, api_key)
                if val is not None:
                    responses.append(val)
                time.sleep(SLEEP_BETWEEN_CALLS)

            if responses:
                mean_resp = np.mean(responses)
                prob = mean_resp / 10.0
            else:
                mean_resp = np.nan
                prob = np.nan

            rows.append({
                "pat_id":         pid,
                "condition":      cond,
                "responses":      str(responses),
                "mean_response":  mean_resp,
                "predicted_prob": prob,
            })
            done += 1

            if len(rows) % 2 == 0:  # save after every patient
                chunk = pd.DataFrame(rows)
                if out_path.exists():
                    chunk.to_csv(out_path, mode="a", header=False, index=False)
                else:
                    chunk.to_csv(out_path, index=False)
                rows = []

            if done % 20 == 0:
                print(f"  Progress: {done}/{total} conditions done")

    if rows:
        chunk = pd.DataFrame(rows)
        if out_path.exists():
            chunk.to_csv(out_path, mode="a", header=False, index=False)
        else:
            chunk.to_csv(out_path, index=False)

    result = pd.read_csv(out_path)
    print(f"\nDone! Saved {len(result)} rows to {out_path}")
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run LLM predictions for H-AIPW analysis")
    parser.add_argument("--model", choices=["claude", "gemini"], required=True,
                        help="Which LLM to use")
    parser.add_argument("--n", default="5",
                        help="Number of patients to process ('all' for full dataset, or an integer)")
    args = parser.parse_args()

    if args.model == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Missing API key!\n"
                "Set it with:  export ANTHROPIC_API_KEY='sk-ant-...'\n"
                "Get one at:   https://console.anthropic.com"
            )
    else:
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Missing API key!\n"
                "Set it with:  export GOOGLE_API_KEY='AIza...'\n"
                "Get one at:   https://aistudio.google.com/app/apikey"
            )

    df = load_ist3_complete_cases()
    n = len(df) if args.n == "all" else int(args.n)
    print(f"\nRunning {args.model.upper()} on {n} patients × 2 conditions × {N_PROMPTS} prompts")
    print(f"= {n * 2 * N_PROMPTS} total API calls\n")

    results = predict(df, args.model, api_key, n)

    print("\n── Summary ──")
    print(results.groupby("condition")["predicted_prob"].describe().round(3))


if __name__ == "__main__":
    main()

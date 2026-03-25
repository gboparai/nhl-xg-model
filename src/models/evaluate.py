"""
evaluate.py
===========
Run the XGModel pipeline, score the holdout seasons, then produce
a per-situation AUC report and save it to results/.

Results are written to a TIMESTAMPED file so they never overwrite
the original baseline in results/baseline_results.txt.

Usage
-----
    python src/models/evaluate.py
    python src/models/evaluate.py --retrain
    python src/models/evaluate.py --tag <label>
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Allow running from the project root or from inside src/models/
HERE    = Path(__file__).parent            # src/models/
ROOT    = HERE.parent.parent               # project root
RESULTS = ROOT / "results"

sys.path.insert(0, str(HERE))   # find xg_model in src/models/
from xg_model import XGModel  # noqa: E402


# ── situation decoder ──────────────────────────────────────────────────────

def decode_situation(code: float, is_home: float = float("nan")) -> str:
    """
    situation_code format: [away_goalie][away_skaters][home_skaters][home_goalie]

    PP/PK is labelled from the SHOOTER's perspective using is_home:
      is_home=1 -> shooting team is home
      is_home=0 -> shooting team is away

    Example:
      1451 (away 4sk, home 5sk = home PP):
        is_home=1 -> home shot -> PP
        is_home=0 -> away shot -> PK (shorthanded)
      1541 (away 5sk, home 4sk = away PP):
        is_home=0 -> away shot -> PP
        is_home=1 -> home shot -> PK (shorthanded)
    """
    try:
        c = int(code)
    except (ValueError, TypeError):
        return "other"

    away_g  = (c // 1000) % 10
    away_sk = (c // 100) % 10
    home_sk = (c // 10) % 10
    home_g  = c % 10

    import math
    ih = -1 if math.isnan(float(is_home)) else int(is_home)

    # Goalie pulled — the team that pulled their goalie is the extra attacker (PP);
    # the opposing team shooting INTO that empty net is EN.
    if away_g == 0 or home_g == 0:
        if ih == -1:
            return "EN"   # unknown shooter, fall back to EN
        if away_g == 0:
            # Away pulled their goalie — away is the extra attacker
            return "PP" if ih == 0 else "EN"
        else:
            # Home pulled their goalie — home is the extra attacker
            return "PP" if ih == 1 else "EN"

    diff = home_sk - away_sk   # >0 = home PP, <0 = away PP

    if diff == 0:
        return "5v5"     # 5v5, 4v4, 3v3 all even-strength

    if ih == -1:
        return "other"
    if diff > 0:   # home has more skaters (home PP)
        return "PP" if ih == 1 else "PK"
    else:          # away has more skaters (away PP)
        return "PP" if ih == 0 else "PK"


# ── per-situation metrics ─────────────────────────────────────────────────

def _calibration(s: pd.DataFrame, xg_col: str, label_col: str) -> float | None:
    goals = s[label_col].sum()
    if goals == 0:
        return None
    return round(s[xg_col].sum() / goals * 100, 1)


def evaluate_by_situation(df: pd.DataFrame, xg_col: str, label_col: str) -> list[dict]:
    non_en = df[df["situation"] != "EN"]
    rows = []

    for sit in ["5v5", "PP", "PK"]:
        s = non_en[non_en["situation"] == sit]
        if len(s) < 10 or s[label_col].nunique() < 2:
            rows.append(dict(situation=sit, n=len(s), goal_pct=None, auc=None, log_loss=None, brier=None, calibration=None))
            continue
        rows.append(dict(
            situation=sit,
            n=len(s),
            goal_pct=s[label_col].mean(),
            auc=roc_auc_score(s[label_col], s[xg_col]),
            log_loss=log_loss(s[label_col], s[xg_col]),
            brier=brier_score_loss(s[label_col], s[xg_col]),
            calibration=_calibration(s, xg_col, label_col),
        ))

    ov = non_en[non_en["situation"].isin(["5v5", "PP", "PK"])]
    if len(ov) > 10:
        rows.append(dict(
            situation="Overall*",
            n=len(ov),
            goal_pct=ov[label_col].mean(),
            auc=roc_auc_score(ov[label_col], ov[xg_col]),
            log_loss=log_loss(ov[label_col], ov[xg_col]),
            brier=brier_score_loss(ov[label_col], ov[xg_col]),
            calibration=_calibration(ov, xg_col, label_col),
        ))

    en = df[df["situation"] == "EN"]
    if len(en) >= 10 and en[label_col].nunique() >= 2:
        rows.append(dict(
            situation="EN",
            n=len(en),
            goal_pct=en[label_col].mean(),
            auc=roc_auc_score(en[label_col], en[xg_col]),
            log_loss=log_loss(en[label_col], en[xg_col]),
            brier=brier_score_loss(en[label_col], en[xg_col]),
            calibration=_calibration(en, xg_col, label_col),
        ))
    else:
        rows.append(dict(
            situation="EN",
            n=len(en),
            goal_pct=en[label_col].mean() if len(en) else None,
            auc=None, log_loss=None, brier=None, calibration=_calibration(en, xg_col, label_col),
        ))
    return rows


# ── display ────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], tag: str, seasons: list) -> None:
    print("=" * 78)
    print(f"  NHL xG MODEL — {tag.upper()}")
    print(f"  Test seasons : {', '.join(str(s) for s in seasons)}")
    print("=" * 78)
    print(f"  {'Situation':<12} {'n':>8}  {'Goal%':>6}  {'AUC':>7}  {'LogLoss':>8}  {'Brier':>7}  {'Calib%':>8}")
    print("-" * 78)
    for r in rows:
        gp  = f"{r['goal_pct']:.3f}" if r["goal_pct"] is not None else "  n/a"
        cal = f"{r['calibration']:.1f}%" if r.get("calibration") is not None else "   n/a"
        if r["auc"] is None:
            if r["situation"] == "Overall*":
                print("-" * 78)
            print(f"  {r['situation']:<12} {r['n']:>8,}  {gp:>6}  {'n/a':>7}  {'n/a':>8}  {'n/a':>7}  {cal:>8}")
        else:
            if r["situation"] == "Overall*":
                print("-" * 78)
            print(f"  {r['situation']:<12} {r['n']:>8,}  {gp:>6}  {r['auc']:>7.4f}  {r['log_loss']:>8.4f}  {r['brier']:>7.4f}  {cal:>8}")
    print()
    print("  * 5v5 + PP + PK combined")
    print("=" * 78)


def build_report(rows: list[dict], tag: str, seasons: list, pred_path: Path) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"NHL xG MODEL — {tag.upper()}",
        f"Generated  : {timestamp}",
        f"Predictions: {pred_path}",
        f"Seasons    : {', '.join(str(s) for s in seasons)}",
        "",
        f"{'Situation':<12} {'n':>8}  {'Goal%':>6}  {'AUC':>7}  {'LogLoss':>8}  {'Brier':>7}  {'Calib%':>8}",
        "-" * 70,
    ]
    for r in rows:
        gp  = f"{r['goal_pct']:.3f}" if r["goal_pct"] is not None else "  n/a"
        cal = f"{r['calibration']:.1f}%" if r.get("calibration") is not None else "   n/a"
        if r["auc"] is None:
            if r["situation"] == "Overall*":
                lines.append("-" * 70)
            lines.append(f"  {r['situation']:<12} {r['n']:>8,}  {gp:>6}  {'n/a':>7}  {'n/a':>8}  {'n/a':>7}  {cal:>8}")
        else:
            if r["situation"] == "Overall*":
                lines.append("-" * 70)
            lines.append(f"  {r['situation']:<12} {r['n']:>8,}  {r['goal_pct']:>6.3f}  {r['auc']:>7.4f}  {r['log_loss']:>8.4f}  {r['brier']:>7.4f}  {cal:>8}")
    lines += [
        "",
        "  * 5v5 + PP + PK combined",
        "",
        "BASELINE REFERENCE (results/baseline_results.txt)",
        "-" * 70,
        "  5v5      AUC: 0.8140  LogLoss: 0.1787  Brier: 0.0484",
        "  PP       AUC: 0.7656  LogLoss: 0.2980  Brier: 0.0876",
        "  PK       AUC: 0.7412  LogLoss: 0.2830  Brier: 0.0814",
        "  Overall* AUC: 0.8059  LogLoss: 0.1988  Brier: 0.0550",
    ]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate NHL xG model")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain from scratch instead of using pretrained pkl")
    parser.add_argument("--tag", default="",
                        help="Short label for this run, included in the output filename")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    parser.add_argument("--data-path", default=None,
                        help="Path to shot CSV to use instead of the default xg_table.csv.gz")
    parser.add_argument("--test-seasons", default=None,
                        help="Comma-separated holdout season IDs, e.g. 20232024,20242025,20252026")
    args = parser.parse_args()

    test_seasons = [s.strip() for s in args.test_seasons.split(",")] if args.test_seasons else None

    runner = XGModel(
        use_pretrained=not args.retrain,
        use_gpu=args.gpu,
        verbose=True,
        data_path=args.data_path,
        test_seasons=test_seasons,
    )
    results = runner.run()

    preds_df = results["predictions"]   # scored holdout shots from the runner

    # Attach situation labels — PP/PK from the shooter's perspective via is_home
    if "situation_code" in preds_df.columns:
        if "is_home" in preds_df.columns:
            preds_df["situation"] = [
                decode_situation(code, ih)
                for code, ih in zip(preds_df["situation_code"], preds_df["is_home"])
            ]
        else:
            preds_df["situation"] = preds_df["situation_code"].map(decode_situation)
    else:
        preds_df["situation"] = "5v5"   # fallback

    seasons = sorted(preds_df["season"].unique())
    label_col = "shot_made" if "shot_made" in preds_df.columns else "is_goal"
    rows = evaluate_by_situation(preds_df, "xg", label_col)

    print_table(rows, args.tag or "evaluation", seasons)

    # ── Save — always timestamped, never overwrites baseline ──────────────
    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"_{args.tag.replace(' ', '-')}" if args.tag else ""
    out_txt = RESULTS / f"xg_model{slug}_{ts}.txt"
    out_csv = RESULTS / f"xg_model{slug}_{ts}.csv"

    out_txt.write_text(build_report(rows, args.tag or "evaluation", seasons, out_txt), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\n  Report saved -> {out_txt}")
    print(f"  CSV    saved -> {out_csv}\n")
    print("  Baseline is in  -> results/baseline_results.txt  (unchanged)\n")


if __name__ == "__main__":
    main()

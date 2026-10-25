#!/usr/bin/env python3
"""Compare each model on native vs translated benchmarks in the same language.

Reads the lm-eval sample logs under results/ (run.sh writes them with
--log_samples) and data/manifest.json (written by prepare.py), and writes
analysis/report.md, analysis/pairs.csv and analysis/multiloko.csv.

Multiple-choice pairs. For each model, accuracy on the native set and on its
translated set, raw and chance-corrected per question ((acc - chance) /
(1 - chance), since option counts differ). The gap is translated minus native,
chance-corrected, with a 95% bootstrap interval that resamples questions within
each set. With two or more models, the report also says whether the native and
the translated set pick the same best model. For Global-MMLU sets it adds the
culturally agnostic (CA) minus culturally sensitive (CS) gap on the annotated
questions.

MultiLoKo. Predictions are scored with MultiLoKo's own eval.py (exact match
after its normalisation). The locality effect per language is EM on locally
sourced questions minus EM on English-sourced questions human-translated into
the language, as MultiLoKo defines it: positive means the translated set
underestimates the model, negative means it overestimates it.

Usage: python analyze.py [--results results] [--reps 2000] [--seed 0]
"""
import argparse
import csv
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_samples(root):
    """-> {task: {model: {question_id: sample}}}, keeping the newest run of each."""
    newest = {}
    for f in Path(root).rglob("samples_lga_*.jsonl"):
        model = f.parent.name
        task = f.name[len("samples_"):].rsplit("_", 1)[0]  # samples_<task>_<timestamp>.jsonl
        if (model, task) not in newest or f.name > newest[(model, task)].name:
            newest[(model, task)] = f
    out = defaultdict(dict)
    for (model, task), f in newest.items():
        rows = (json.loads(line) for line in f.open(encoding="utf-8"))
        out[task][model] = {r["doc"]["id"]: r for r in rows}
    return out


def corrected(sample):
    chance = sample["doc"]["chance"]
    return (float(sample["acc"]) - chance) / (1 - chance)


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap_means(values, reps, rng, chunk=100):
    """Means of `reps` resamples (with replacement) of `values`, drawn in chunks
    so that a 35,000-question set never needs a reps x n matrix in memory."""
    values = np.asarray(values, dtype=float)
    means = []
    for start in range(0, reps, chunk):
        idx = rng.integers(0, len(values), size=(min(chunk, reps - start), len(values)))
        means.append(values[idx].mean(axis=1))
    return np.concatenate(means)


def pct(x):
    return f"{100 * x:.1f}"


def spearman(x, y):
    """Spearman's rho with average ranks for ties; nan if either side is constant."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def mc_pairs(manifest, samples, reps, rng):
    rows, lines = [], []
    for pair in manifest["pairs"]:
        native = manifest["sets"][pair["native"]]
        for t_name in pair["translated"]:
            translated = manifest["sets"][t_name]
            n_runs, t_runs = samples.get(native["task"], {}), samples.get(translated["task"], {})
            models = sorted(set(n_runs) & set(t_runs))
            if not models:
                continue
            lines.append(f"\n### {native['benchmark']} vs {translated['benchmark']} ({pair['language']})\n")
            lines.append("| Model | Native acc | Translated acc | Native (chance-corr.) | Translated (chance-corr.) "
                         "| Gap, translated − native [95% CI] |")
            lines.append("|---|---|---|---|---|---|")
            best = {}
            for m in models:
                nat, tra = list(n_runs[m].values()), list(t_runs[m].values())
                n_acc, t_acc = np.mean([float(s["acc"]) for s in nat]), np.mean([float(s["acc"]) for s in tra])
                n_cor, t_cor = np.mean([corrected(s) for s in nat]), np.mean([corrected(s) for s in tra])
                reps_gap = (bootstrap_means([corrected(s) for s in tra], reps, rng)
                            - bootstrap_means([corrected(s) for s in nat], reps, rng))
                lo, hi = ci(reps_gap)
                gap = t_cor - n_cor
                best[m] = (n_cor, t_cor)
                lines.append(f"| {m} | {pct(n_acc)} (n={len(nat)}) | {pct(t_acc)} (n={len(tra)}) | {pct(n_cor)} | "
                             f"{pct(t_cor)} | {100 * gap:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
                rows.append({"pair": pair["pair"], "language": pair["language"], "model": m,
                             "native_set": pair["native"], "translated_set": t_name,
                             "native_n": len(nat), "translated_n": len(tra),
                             "native_acc": n_acc, "translated_acc": t_acc,
                             "native_corrected": n_cor, "translated_corrected": t_cor,
                             "gap_corrected": gap, "gap_ci_low": lo, "gap_ci_high": hi,
                             "gap_raw": t_acc - n_acc})
            if len(models) >= 2:
                top_n = max(models, key=lambda m: best[m][0])
                top_t = max(models, key=lambda m: best[m][1])
                verdict = "the same model" if top_n == top_t else "**different models**"
                lines.append(f"\nBest on the native set: {top_n}; best on the translated set: {top_t} ({verdict}).")
    return rows, lines


def cs_ca(manifest, samples):
    lines = []
    for name, meta in sorted(manifest["sets"].items()):
        if meta["benchmark"] != "Global-MMLU":
            continue
        for m, runs in sorted(samples.get(meta["task"], {}).items()):
            cs = [float(s["acc"]) for s in runs.values() if s["doc"].get("cs_label") == "CS"]
            ca = [float(s["acc"]) for s in runs.values() if s["doc"].get("cs_label") == "CA"]
            if cs and ca:
                lines.append(f"| {meta['language']} | {m} | {pct(np.mean(ca))} (n={len(ca)}) | "
                             f"{pct(np.mean(cs))} (n={len(cs)}) | {100 * (np.mean(ca) - np.mean(cs)):+.1f} |")
    if lines:
        lines = ["\n## Global-MMLU: culturally agnostic vs culturally sensitive questions\n",
                 "| Language | Model | CA acc | CS acc | CA − CS |", "|---|---|---|---|---|"] + lines
    return lines


def multiloko(manifest, samples, reps, rng):
    info = manifest.get("multiloko") or {}
    ml_sets = {n: s for n, s in manifest["sets"].items() if s["benchmark"] == "MultiLoKo"}
    if not ml_sets or not info:
        return [], []
    models = sorted({m for s in ml_sets.values() for m in samples.get(s["task"], {})})
    rows, lines = [], ["\n## MultiLoKo: locally sourced vs English-sourced, human-translated questions\n"]
    for m in models:
        with tempfile.TemporaryDirectory() as tmp:
            preds, scores = Path(tmp) / "predictions.jsonl", Path(tmp) / "scores.json"
            with preds.open("w", encoding="utf-8") as f:
                for meta in ml_sets.values():
                    for s in samples.get(meta["task"], {}).get(m, {}).values():
                        text = s["filtered_resps"][0] if s.get("filtered_resps") else ""
                        f.write(json.dumps({"language": s["doc"]["multiloko_key"], "id": s["doc"]["id"],
                                            "prediction": text if isinstance(text, str) else str(text)},
                                           ensure_ascii=False) + "\n")
            subprocess.run([sys.executable, str(Path(info["repo"]) / "eval.py"), "-d", info["data"],
                            "-s", info["subset"], "-p", str(preds), "-o", str(scores)], check=True)
            result = json.loads(scores.read_text(encoding="utf-8"))

        def em(key):
            per = [v["em"] for k, v in result.get(key, {}).items() if isinstance(v, dict) and "em" in v]
            return per

        per_lang = []
        for language in sorted({s["language"] for s in ml_sets.values()}):
            local = em(language)
            human = em(f"english_translated_human_{language}")
            if not local or not human:
                continue
            le = np.mean(local) - np.mean(human)
            lo, hi = ci(bootstrap_means(local, reps, rng) - bootstrap_means(human, reps, rng))
            row = {"model": m, "language": language, "em_local": np.mean(local), "em_translated_human": np.mean(human),
                   "locality_effect": le, "le_ci_low": lo, "le_ci_high": hi, "n_local": len(local), "n_human": len(human)}
            rows.append(row)
            per_lang.append(row)
        if not per_lang:
            continue
        les = np.array([r["locality_effect"] for r in per_lang])
        loc = np.array([r["em_local"] for r in per_lang])
        hum = np.array([r["em_translated_human"] for r in per_lang])
        rho = spearman(loc, hum) if len(per_lang) > 2 else float("nan")
        lines.append(f"\n### {m}\n")
        lines.append(f"- Languages: {len(per_lang)}. Mean EM: local {pct(loc.mean())}, translated {pct(hum.mean())}.")
        lines.append(f"- Locality effect (local − translated): mean {100 * les.mean():+.1f}, mean absolute "
                     f"{100 * np.abs(les).mean():.1f}, range {100 * les.min():+.1f} to {100 * les.max():+.1f} points; "
                     f"{int((les > 0).sum())} languages positive (translation underestimates), "
                     f"{int((les < 0).sum())} negative (translation overestimates); "
                     f"{int((np.abs(les) >= 0.10).sum())} with |effect| ≥ 10 points.")
        lines.append(f"- Rank correlation of language difficulty, local vs translated: {rho:.2f}.\n")
        lines.append("| Language | Local EM | Translated EM | Locality effect [95% CI] |")
        lines.append("|---|---|---|---|")
        for r in sorted(per_lang, key=lambda r: r["locality_effect"]):
            lines.append(f"| {r['language']} | {pct(r['em_local'])} | {pct(r['em_translated_human'])} | "
                         f"{100 * r['locality_effect']:+.1f} [{100 * r['le_ci_low']:+.1f}, {100 * r['le_ci_high']:+.1f}] |")
    return rows, lines


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.json"))
    ap.add_argument("--out", default=str(HERE / "analysis"))
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = load_samples(args.results)
    if not samples:
        sys.exit(f"no samples_lga_*.jsonl under {args.results}: run ./run.sh first")
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    pair_rows, pair_lines = mc_pairs(manifest, samples, args.reps, rng)
    ml_rows, ml_lines = multiloko(manifest, samples, args.reps, rng)
    report = ["# Native vs translated: same models, same harness, same prompt\n",
              "Gaps are translated minus native, in chance-corrected points; intervals are 95% bootstrap "
              "intervals over questions. Native and translated sets differ in content and difficulty, so a "
              "raw gap alone is not evidence of over- or understatement: read it together with the best-model "
              "comparison and the MultiLoKo locality effect, which holds the benchmark fixed.\n",
              "## Multiple-choice pairs"] + pair_lines + cs_ca(manifest, samples) + ml_lines
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_csv(out / "pairs.csv", pair_rows)
    write_csv(out / "multiloko.csv", ml_rows)
    print(f"wrote {out / 'report.md'} ({len(pair_rows)} pair rows, {len(ml_rows)} MultiLoKo rows)")


if __name__ == "__main__":
    main()

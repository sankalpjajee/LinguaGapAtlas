#!/usr/bin/env python3
"""Compare each model on native vs translated benchmarks in the same language.

Reads the lm-eval sample logs under results/ (run.sh writes them with
--log_samples) and data/manifest.json (written by prepare.py), and writes
analysis/report.md, pairs.csv, comparisons.csv and multiloko.csv.

Multiple-choice pairs. For each model, accuracy on the native set and on its
translated set, raw and chance-corrected per question ((acc - chance) /
(1 - chance), since option counts differ). The gap is translated minus native,
chance-corrected, with a 95% bootstrap interval that resamples questions within
each set. With two or more models, each pair of models is compared on both sets:
the difference between them (paired over questions, since both answer the same
ones) and how that difference changes from the native to the translated set,
each with a 95% bootstrap interval. A ranking counts as reversed only when the
difference is significant on both sets. For Global-MMLU sets the report adds
the culturally agnostic (CA) minus culturally sensitive (CS) gap on the
annotated questions.

MultiLoKo. Predictions are scored with MultiLoKo's own eval.py (exact match
after its normalisation), against a copy of its reference answers in which
answer lists stored as one semicolon-joined string are split (see
multiloko_reference). The locality effect per language is EM on locally
sourced questions minus EM on English-sourced questions human-translated into
the language, as MultiLoKo defines it: positive means the translated set
underestimates the model, negative means it overestimates it.

Usage: python analyze.py [--results results] [--reps 2000] [--seed 0]
"""
import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEMICOLON = re.compile(r"[;；]")


def load_samples(root):
    """-> {task: {model: {question_id: sample}}}. For each model and task, keeps the
    newest complete run. A --limit run (a smoke test) is used only when there is no
    complete run, so a pilot never replaces full results, whenever it ran."""
    limited, best = {}, {}
    for f in Path(root).rglob("samples_lga_*.jsonl"):
        model = f.parent.name
        task, stamp = f.name[len("samples_"):-len(".jsonl")].rsplit("_", 1)  # samples_<task>_<stamp>.jsonl
        if (f.parent, stamp) not in limited:  # the run's results_<stamp>.json records its --limit
            run = f.parent / f"results_{stamp}.json"
            config = json.loads(run.read_text(encoding="utf-8")).get("config", {}) if run.exists() else {}
            limited[(f.parent, stamp)] = config.get("limit") is not None
        rank = (not limited[(f.parent, stamp)], stamp)
        if (model, task) not in best or rank > best[(model, task)][0]:
            best[(model, task)] = (rank, f)
    out = defaultdict(dict)
    for (model, task), (_, f) in best.items():
        rows = (json.loads(line) for line in f.open(encoding="utf-8"))
        out[task][model] = {r["doc"]["id"]: r for r in rows}
    pilots = sorted(f"{model}/{task}" for (model, task), ((complete, _), _) in best.items() if not complete)
    if pilots:
        print(f"warning: {len(pilots)} sets have only --limit (smoke-test) runs, e.g. {pilots[0]}")
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


def paired_difference(runs, a, b):
    """Per question, model a's chance-corrected score minus model b's, on the questions both answered."""
    return [corrected(runs[a][q]) - corrected(runs[b][q]) for q in sorted(set(runs[a]) & set(runs[b]))]


def mc_pairs(manifest, samples, reps, rng):
    rows, comparisons, lines = [], [], []
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
            for m in models:
                nat, tra = list(n_runs[m].values()), list(t_runs[m].values())
                n_acc, t_acc = np.mean([float(s["acc"]) for s in nat]), np.mean([float(s["acc"]) for s in tra])
                n_cor, t_cor = np.mean([corrected(s) for s in nat]), np.mean([corrected(s) for s in tra])
                reps_gap = (bootstrap_means([corrected(s) for s in tra], reps, rng)
                            - bootstrap_means([corrected(s) for s in nat], reps, rng))
                lo, hi = ci(reps_gap)
                gap = t_cor - n_cor
                lines.append(f"| {m} | {pct(n_acc)} (n={len(nat)}) | {pct(t_acc)} (n={len(tra)}) | {pct(n_cor)} | "
                             f"{pct(t_cor)} | {100 * gap:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
                rows.append({"pair": pair["pair"], "language": pair["language"], "model": m,
                             "native_set": pair["native"], "translated_set": t_name,
                             "native_n": len(nat), "translated_n": len(tra),
                             "native_acc": n_acc, "translated_acc": t_acc,
                             "native_corrected": n_cor, "translated_corrected": t_cor,
                             "gap_corrected": gap, "gap_ci_low": lo, "gap_ci_high": hi,
                             "gap_raw": t_acc - n_acc})
            if len(models) < 2:
                continue
            lines.append("\n| Models, ahead on the native set first | Native difference [95% CI] "
                         "| Translated difference [95% CI] | Change [95% CI] |")
            lines.append("|---|---|---|---|")
            verdicts = []
            for a, b in itertools.combinations(models, 2):
                dn, dt = paired_difference(n_runs, a, b), paired_difference(t_runs, a, b)
                if np.mean(dn) < 0:  # name first the model that is ahead on the native set
                    a, b, dn, dt = b, a, [-d for d in dn], [-d for d in dt]
                bn, bt = bootstrap_means(dn, reps, rng), bootstrap_means(dt, reps, rng)
                (n_lo, n_hi), (t_lo, t_hi), (c_lo, c_hi) = ci(bn), ci(bt), ci(bt - bn)
                d_n, d_t = float(np.mean(dn)), float(np.mean(dt))
                if d_t < 0 and n_lo > 0 and t_hi < 0:
                    verdict, text = "ranking reverses", "**the ranking reverses**, significantly on both sets"
                elif d_t < 0:
                    verdict, text = "reverses, not significantly", "the point estimates reverse, but not significantly on both sets"
                else:
                    verdict, text = "same ranking", "same ranking on both sets"
                changed = c_lo > 0 or c_hi < 0
                lines.append(f"| {a} − {b} | {100 * d_n:+.1f} [{100 * n_lo:+.1f}, {100 * n_hi:+.1f}] | "
                             f"{100 * d_t:+.1f} [{100 * t_lo:+.1f}, {100 * t_hi:+.1f}] | "
                             f"{100 * (d_t - d_n):+.1f} [{100 * c_lo:+.1f}, {100 * c_hi:+.1f}] |")
                verdicts.append(f"{a} vs {b}: {text}; the difference between them changes by {100 * (d_t - d_n):+.1f} "
                                f"points, {'a significant change' if changed else 'not significant'}.")
                comparisons.append({"pair": pair["pair"], "language": pair["language"], "native_set": pair["native"],
                                    "translated_set": t_name, "ahead_on_native": a, "other": b,
                                    "native_difference": d_n, "native_ci_low": n_lo, "native_ci_high": n_hi,
                                    "translated_difference": d_t, "translated_ci_low": t_lo, "translated_ci_high": t_hi,
                                    "change": d_t - d_n, "change_ci_low": c_lo, "change_ci_high": c_hi,
                                    "verdict": verdict, "change_significant": changed})
            lines.append("")
            lines.extend(verdicts)
    return rows, comparisons, lines


def summary(pair_rows, comparisons):
    lines = ["## Summary\n"]
    if pair_rows:
        lower = sum(r["gap_ci_high"] < 0 for r in pair_rows)
        higher = sum(r["gap_ci_low"] > 0 for r in pair_rows)
        lines.append(f"- Translated minus native, per model and pair: {lower + higher} of {len(pair_rows)} gaps are "
                     f"significant, {lower} lower on the translated set and {higher} higher.")
    if comparisons:
        reversed_ = sum(c["verdict"] == "ranking reverses" for c in comparisons)
        weak = sum(c["verdict"] == "reverses, not significantly" for c in comparisons)
        changed = sum(c["change_significant"] for c in comparisons)
        lines.append(f"- Comparing models: the translated set reverses the ranking in {reversed_} of {len(comparisons)} "
                     f"comparisons, significantly on both sets ({weak} more reverse in point estimates only), and "
                     f"significantly changes the difference between the models in {changed}.")
    lines.append("- Each interval is a 95% bootstrap interval for one comparison, not corrected for multiple comparisons.\n")
    return lines


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
                 "Most culturally sensitive questions come from the humanities and social sciences, which models find "
                 "easier than the STEM-heavy agnostic ones, so this gap mixes culture with subject. Global-MMLU's "
                 "authors find the same and attribute it to the subject mix.\n",
                 "| Language | Model | CA acc | CS acc | CA − CS |", "|---|---|---|---|---|"] + lines
    return lines


def multiloko_reference(src, dst, subset):
    """Copies the reference answers MultiLoKo's eval.py reads (<language>/<subset>.jsonl
    and <language>/<subset>_translated*.jsonl) from src to dst, splitting answer
    lists stored as one semicolon-joined string. At the pinned commit, 158 of the
    250 simplified Mandarin human translations store their answers that way, and
    as one string no prediction can match them. Returns {eval.py set key: number
    of questions split}."""
    split = Counter()
    for f in sorted(Path(src).glob(f"*/{subset}.jsonl")) + sorted(Path(src).glob(f"*/{subset}_translated*.jsonl")):
        key = f.stem.replace(subset, f.parent.name)  # eval.py's name for the set
        out = Path(dst) / f.parent.name / f.name
        out.parent.mkdir(parents=True, exist_ok=True)
        with f.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                if any(SEMICOLON.search(t) for t in row.get("targets", [])):
                    row["targets"] = [p.strip() for t in row["targets"] for p in SEMICOLON.split(t) if p.strip()]
                    split[key] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return split


def multiloko(manifest, samples, reps, rng):
    info = manifest.get("multiloko") or {}
    ml_sets = {n: s for n, s in manifest["sets"].items() if s["benchmark"] == "MultiLoKo"}
    if not ml_sets or not info:
        return [], []
    models = sorted({m for s in ml_sets.values() for m in samples.get(s["task"], {})})
    rows, lines = [], ["\n## MultiLoKo: locally sourced vs English-sourced, human-translated questions\n"]
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        reference = Path(tmp) / "reference"
        split = multiloko_reference(info["data"], reference, info["subset"])
        for m in models:
            preds, scores = Path(tmp) / "predictions.jsonl", Path(tmp) / "scores.json"
            with preds.open("w", encoding="utf-8") as f:
                for meta in ml_sets.values():
                    for s in samples.get(meta["task"], {}).get(m, {}).values():
                        text = s["filtered_resps"][0] if s.get("filtered_resps") else ""
                        f.write(json.dumps({"language": s["doc"]["multiloko_key"], "id": s["doc"]["id"],
                                            "prediction": text if isinstance(text, str) else str(text)},
                                           ensure_ascii=False) + "\n")
            subprocess.run([sys.executable, str(Path(info["repo"]) / "eval.py"), "-d", str(reference),
                            "-s", info["subset"], "-p", str(preds), "-o", str(scores)], check=True)
            results[m] = json.loads(scores.read_text(encoding="utf-8"))
    split = Counter({k: n for k, n in split.items() if k in {s["multiloko_key"] for s in ml_sets.values()}})
    if split:
        (top, n_top), = split.most_common(1)
        lines.append(f"In MultiLoKo's data, {sum(split.values())} of the questions scored here store their accepted "
                     f"answers as one semicolon-joined string ({n_top} of them in {top}). No answer could match such "
                     "a string, so it is split into separate answers before scoring.")
    for m, result in results.items():

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


TABLE_LABELS = {"seaexam_native_id": "SeaExam", "parsinlu_translated_fa": "ParsiNLU (MT)"}  # short names


def model_names(path):
    """{lm-eval results directory name: short name} from models.tsv, in file order:
    "Aya Expanse 8B (instruction-tuned, 23 languages)" becomes "Aya Expanse 8B"."""
    names = {}
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
            if line.strip() and not line.startswith("#"):
                model, label = (line.split("\t") + [""])[:2]
                key = re.sub(r"[\"<>:/|\\?*\[\]]+", "__", model.strip())  # as lm-eval names the directory
                names[key] = label.split(" (")[0].strip() or model.strip()
    return names


def tex(text):
    return re.sub(r"([&%$#_{}])", r"\\\1", str(text))


def tex_interval(x, lo, hi):
    return f"${100 * x:+.1f}$ {{\\scriptsize[${100 * lo:+.1f}$,\\,${100 * hi:+.1f}$]}}"


def latex_mc(manifest, pair_rows, comparisons, names):
    """Table: per pair, each model's chance-corrected native and translated score and
    the gap; with two models, also the difference between them on each set and how
    it changes. Needs booktabs (acmart loads it)."""
    present = {r["model"] for r in pair_rows}
    models = [m for m in names if m in present] + sorted(present - set(names))
    name = lambda m: tex(names.get(m, m))
    by = {(r["native_set"], r["translated_set"], r["model"]): r for r in pair_rows}
    pairs = list(dict.fromkeys((r["native_set"], r["translated_set"], r["language"]) for r in pair_rows))
    label = lambda s: tex(TABLE_LABELS.get(s, manifest["sets"][s]["benchmark"].split(" (")[0]))
    # Name only the native set; mark the rows whose translated set is not the usual one.
    usual = Counter(manifest["sets"][t]["benchmark"] for _, t, _ in pairs).most_common(1)[0][0]
    unusual = {t for _, t, _ in pairs if manifest["sets"][t]["benchmark"] != usual}
    two = len(models) == 2
    if two:  # orient the difference so that it is positive in most native sets
        first = models[0] if 2 * sum(c["ahead_on_native"] == models[0] for c in comparisons) >= len(comparisons) else models[1]
        second = models[1] if first == models[0] else models[0]
    cols = "ll" + "rrr" * len(models) + ("rrr" if two else "")
    out = ["% Written by analyze.py from the runs under results/; regenerate rather than edit.",
           "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           f"\\begin{{tabular}}{{{cols}}}", "\\toprule"]
    head = " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{name(m)}}}" for m in models)
    rules = "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(len(models) + two))
    sub = "Lang. & Native set" + " & Native & Transl. & Gap" * len(models)
    if two:
        head += " & \\multicolumn{3}{c}{Difference}"
        sub += " & Native & Transl. & Change"
    out += [head + " \\\\", rules, sub + " \\\\", "\\midrule"]
    for nat, tra, lang in pairs:
        cells = [lang, label(nat) + ("$^\\dagger$" if tra in unusual else "")]
        for m in models:
            r = by.get((nat, tra, m))
            cells += ([f"${100 * r['native_corrected']:.1f}$", f"${100 * r['translated_corrected']:.1f}$",  # may be < 0
                       tex_interval(r["gap_corrected"], r["gap_ci_low"], r["gap_ci_high"])] if r else ["--"] * 3)
        if two:
            c = next((c for c in comparisons if (c["native_set"], c["translated_set"]) == (nat, tra)
                      and {c["ahead_on_native"], c["other"]} == {first, second}), None)
            if c:
                s = 1 if c["ahead_on_native"] == first else -1
                starred = lambda x, lo, hi: f"${100 * s * x:+.1f}{'^*' if lo > 0 or hi < 0 else ''}$"
                lo, hi = sorted((s * c["change_ci_low"], s * c["change_ci_high"]))
                cells += [starred(c["native_difference"], c["native_ci_low"], c["native_ci_high"]),
                          starred(c["translated_difference"], c["translated_ci_low"], c["translated_ci_high"]),
                          tex_interval(s * c["change"], lo, hi)]
            else:
                cells += ["--"] * 3
        out.append(" & ".join(cells) + " \\\\")
    caption = ("Native and translated benchmarks in the same language, scored with the same harness and prompt. "
               "Scores are chance-corrected accuracy, $(\\mathit{acc}-\\mathit{chance})/(1-\\mathit{chance})$, in "
               "points; gap is translated minus native")
    if two:
        caption += (f". Difference: how far {name(first)} is ahead of {name(second)} on each set "
                    "($^*$: the 95\\% interval excludes zero), and how much that changes from the native to the "
                    "translated set")
    caption += f". Brackets: 95\\% bootstrap intervals over questions. The translated set is {tex(usual)} in the same language"
    if unusual:
        caption += "; for $^\\dagger$, " + ", ".join(sorted(label(t) for t in unusual))
    caption += (". Global-MMLU sets are human translations except ar (machine translation post-edited by "
                "professionals) and tr (machine translation, partly edited by volunteers). SeaExam: its native "
                "Indonesian exams, standing in for SEACrowd. ParsiNLU (MT): ParsiNLU entailment items "
                "machine-translated from MNLI.")
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", "\\label{tab:native-vs-translated}",
            "\\end{table*}"]
    return "\n".join(out) + "\n"


def latex_multiloko(ml_rows, names):
    """Table: per language and model, MultiLoKo exact match on local and on
    translated questions and the locality effect, with summary rows."""
    present = {r["model"] for r in ml_rows}
    models = [m for m in names if m in present] + sorted(present - set(names))
    by = {(r["model"], r["language"]): r for r in ml_rows}
    out = ["% Written by analyze.py from the runs under results/; regenerate rather than edit.",
           "\\begin{table*}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
           f"\\begin{{tabular}}{{l{'rrr' * len(models)}}}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{tex(names.get(m, m))}}}" for m in models) + " \\\\",
           "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(models))),
           "Language" + " & Local & Transl. & Effect" * len(models) + " \\\\", "\\midrule"]
    for language in sorted({r["language"] for r in ml_rows}):
        cells = [tex(language.replace("_", " ").title())]
        for m in models:
            r = by.get((m, language))
            cells += ([f"{100 * r['em_local']:.1f}", f"{100 * r['em_translated_human']:.1f}",
                       tex_interval(r["locality_effect"], r["le_ci_low"], r["le_ci_high"])] if r else ["--"] * 3)
        out.append(" & ".join(cells) + " \\\\")
    out.append("\\midrule")
    mean, absolute, rho = ["Mean"], ["Mean $|$effect$|$"], ["Spearman $\\rho$, local vs transl."]
    for m in models:
        rows = [r for r in ml_rows if r["model"] == m]
        les = np.array([r["locality_effect"] for r in rows])
        loc, hum = [r["em_local"] for r in rows], [r["em_translated_human"] for r in rows]
        mean += [f"{100 * np.mean(loc):.1f}", f"{100 * np.mean(hum):.1f}", f"${100 * les.mean():+.1f}$"]
        absolute += ["", "", f"{100 * np.abs(les).mean():.1f}"]
        rho += ["", "", f"{spearman(loc, hum):.2f}" if len(rows) > 2 else "--"]
    out += [" & ".join(row) + " \\\\" for row in (mean, absolute, rho)]
    caption = ("MultiLoKo exact match (\\%) on locally sourced questions and on English-sourced questions "
               "human-translated into the same language, scored with MultiLoKo's own evaluation script. Effect is "
               "local minus translated, MultiLoKo's locality effect: positive means the translated set underestimates "
               "the model. Brackets: 95\\% bootstrap intervals over questions. Answer lists that MultiLoKo's data "
               "stores as one semicolon-joined string, mostly in the simplified Mandarin translations, were split "
               "before scoring.")
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{caption}}}", "\\label{tab:multiloko}", "\\end{table*}"]
    return "\n".join(out) + "\n"


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
    ap.add_argument("--models", default=str(HERE / "models.tsv"), help="for the model names in the LaTeX tables")
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
    pair_rows, comparisons, pair_lines = mc_pairs(manifest, samples, args.reps, rng)
    ml_rows, ml_lines = multiloko(manifest, samples, args.reps, rng)
    report = ["# Native vs translated: same models, same harness, same prompt\n",
              "Gaps are translated minus native, in chance-corrected points; intervals are 95% bootstrap "
              "intervals over questions. Native and translated sets differ in content and difficulty, so a "
              "raw gap alone is not evidence of over- or understatement: read it together with the model "
              "comparison, which asks whether the translated set changes how the models compare, and with the "
              "MultiLoKo locality effect, which holds the benchmark fixed.\n"]
    report += summary(pair_rows, comparisons) + ["## Multiple-choice pairs"] + pair_lines
    report += cs_ca(manifest, samples) + ml_lines
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_csv(out / "pairs.csv", pair_rows)
    write_csv(out / "comparisons.csv", comparisons)
    write_csv(out / "multiloko.csv", ml_rows)
    names = model_names(args.models)
    if pair_rows:
        (out / "table_mc.tex").write_text(latex_mc(manifest, pair_rows, comparisons, names), encoding="utf-8")
    if ml_rows:
        (out / "table_multiloko.tex").write_text(latex_multiloko(ml_rows, names), encoding="utf-8")
    print(f"wrote {out / 'report.md'} ({len(pair_rows)} pair rows, {len(comparisons)} model comparisons, "
          f"{len(ml_rows)} MultiLoKo rows)")


if __name__ == "__main__":
    main()

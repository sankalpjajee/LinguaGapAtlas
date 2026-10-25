#!/usr/bin/env python3
"""Build the native-vs-translated evaluation: download each native, culturally
grounded benchmark in the LinguaGapAtlas corpus and its closest translated
equivalent in the same language, normalise both to one schema, and write the
lm-eval task files that score them with one prompt format.

Pairs (native -> translated, same language):
  ArabicMMLU ar        -> Global-MMLU ar
  CMMLU zh             -> Global-MMLU zh
  TurkishMMLU tr       -> Global-MMLU tr
  KMMLU ko             -> Global-MMLU ko
  M3Exam zh/it/pt/vi/sw -> Global-MMLU (same language); zh/vi/th also -> SeaExam's
                          Google-translated MMLU; af -> MMLU-ProX af (10 options)
  FarsTail fa (NLI)    -> ParsiNLU entailment, MNLI machine-translated into Persian
  MultiLoKo, 30 langs  -> MultiLoKo's English-sourced questions translated into the
                          same language (human and machine translations)
  SeaExam id (native Indonesian exams) -> SeaExam Google-translated MMLU id: a
                          Southeast Asian stand-in, because SEACrowd is a data hub
                          with no single test set and cannot be run as one benchmark

Writes data/ (normalised JSONL, never to be redistributed), tasks/ (lm-eval YAML)
and data/manifest.json (the pairs analyze.py compares). External repositories go
in external/. Usage:

  python prepare.py                 # everything
  python prepare.py --cap 3000      # at most 3,000 test questions per set
  python prepare.py --only m3exam,farstail --skip-multiloko
"""
import argparse
import ast
import hashlib
import csv
import importlib.util
import io
import json
import random
import re
import subprocess
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
DATA, TASKS, EXTERNAL = HERE / "data", HERE / "tasks", HERE / "external"
LETTERS = "ABCDEFGHIJ"

# Pinned sources outside the Hugging Face Hub (commit hashes checked 2026-10-25).
MULTILOKO_REPO = "https://github.com/facebookresearch/multiloko"
MULTILOKO_COMMIT = "a2b9c2fb894710d439259a67ebe8a05b64fcfa81"
MULTILOKO_PASSWORD = "multiloko"  # published in the MultiLoKo README; please do not mirror the data
FARSTAIL_RAW = "https://raw.githubusercontent.com/dml-qom/FarsTail/5d6322c40368ea5fab50e406e083e537a7d97817/data/{}"
M3EXAM_ZIP_URL = "https://drive.usercontent.google.com/download?id=1eREETRklmXJLXrNPTyHxQ3RFdPhq_Nes&export=download&confirm=t"
M3EXAM_PASSWORD = b"12317"  # published in the M3Exam README
M3EXAM_SHA256 = "0f0d5477f7a7138c2da1078d23e09b1f563eaa3acf196944dcd668b0d3b16821"
PARSINLU_PARQUET = ("https://huggingface.co/datasets/persiannlp/parsinlu_entailment/resolve/"
                    "refs%2Fconvert%2Fparquet/parsinlu-repo/{}/0000.parquet")

NLI_CHOICES = ["entailment", "neutral", "contradiction"]
NLI_LABEL = {"e": 0, "n": 1, "c": 2}


def item(question, choices, gold, subject, **extra):
    choices = [str(c).strip() for c in choices]
    if not (2 <= len(choices) <= len(LETTERS)) or not (0 <= gold < len(choices)):
        raise ValueError(f"bad item: {question[:60]!r} {choices} {gold}")
    return {"question": str(question).strip(), "choices": choices, "letters": list(LETTERS[:len(choices)]),
            "gold": gold, "chance": 1 / len(choices), "subject": subject, **extra}


def nli_item(premise, hypothesis, label, subject):
    question = f"Premise: {premise.strip()}\nHypothesis: {hypothesis.strip()}\nWhat is the relation of the hypothesis to the premise?"
    return item(question, NLI_CHOICES, NLI_LABEL[label], subject)


def as_list(value):
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


# ---------------------------------------------------------------- native exams

def arabicmmlu(split):
    out = []
    for r in load_dataset("MBZUAI/ArabicMMLU", "All", split=split):
        options = [r[f"Option {i}"] for i in range(1, 6) if r[f"Option {i}"] not in (None, "")]
        question = r["Question"] if not r["Context"] else f"{r['Context']}\n\n{r['Question']}"
        out.append(item(question, options, LETTERS.index(r["Answer Key"].strip()), r["Subject"]))
    return out


def cmmlu(split):
    # The Hub copy is script-based, which current `datasets` cannot load; read the
    # official CSV archive from the same repository instead.
    path = hf_hub_download("haonan-li/cmmlu", "cmmlu_v1_0_1.zip", repo_type="dataset")
    out = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(n for n in z.namelist() if n.startswith(f"{split}/") and n.endswith(".csv")):
            with z.open(name) as f:
                for r in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                    out.append(item(r["Question"], [r[k] for k in "ABCD"], LETTERS.index(r["Answer"].strip()),
                                    Path(name).stem))
    return out


def kmmlu(split):
    out = []
    for config in get_dataset_config_names("HAERAE-HUB/KMMLU"):
        for r in load_dataset("HAERAE-HUB/KMMLU", config, split=split):
            out.append(item(r["question"], [r[k] for k in "ABCD"], int(r["answer"]) - 1, config))
    return out


def turkishmmlu(split):
    if split == "test":  # the public test subset; answers are 0-based indices
        return [item(r["question"], as_list(r["choices"]), int(r["answer"]), r["subject"])
                for r in load_dataset("AYueksel/TurkishMMLU", "All", split="test")]
    out = []  # dev sets live in the per-subject configs, with letter answers
    for config in get_dataset_config_names("AYueksel/TurkishMMLU"):
        if config == "All":
            continue
        for r in load_dataset("AYueksel/TurkishMMLU", config, split="dev"):
            out.append(item(r["question"], as_list(r["choices"]), LETTERS.index(str(r["answer"]).strip()), config))
    return out


M3EXAM_LANGS = {"zh": "chinese", "it": "italian", "pt": "portuguese", "vi": "vietnamese",
                "th": "thai", "sw": "swahili", "af": "afrikaans"}
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
M3EXAM_LABEL = re.compile(r"^\s*[\(（]?\s*([A-Za-z]|\d{1,2})\s*[\)）.．、:：]\s*")


def m3exam_zip(path_arg):
    path = Path(path_arg) if path_arg else EXTERNAL / "m3exam-data.zip"
    if not path.exists():
        EXTERNAL.mkdir(exist_ok=True)
        print(f"downloading M3Exam ({M3EXAM_ZIP_URL}) -> {path}")
        urllib.request.urlretrieve(M3EXAM_ZIP_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != M3EXAM_SHA256:
        print(f"warning: {path} sha256 {digest} differs from the archive these scripts were tested on")
    return zipfile.ZipFile(path)


def m3exam(lang, split, archive):
    """Text-only M3Exam questions. Option labels ("(1)", "A.", "a)", Thai digits)
    are stripped and the answer mapped to the option it names."""
    rows = json.loads(archive.read(f"data/text-question/{M3EXAM_LANGS[lang]}-questions-{split}.json",
                                   pwd=M3EXAM_PASSWORD))
    out, skipped = [], 0
    for r in rows:
        options, labels = [], []
        for o in as_list(r["options"]):
            o = str(o).translate(THAI_DIGITS)
            m = M3EXAM_LABEL.match(o)
            labels.append(m.group(1).lower() if m else None)
            options.append(o[m.end():] if m else o)
        answer = str(r["answer_text"]).strip().translate(THAI_DIGITS).lower().strip("().")
        if answer in labels:
            gold = labels.index(answer)
        elif answer.isdigit() and 1 <= int(answer) <= len(options):
            gold = int(answer) - 1
        elif len(answer) == 1 and answer.isalpha() and ord(answer) - 97 < len(options):
            gold = ord(answer) - 97
        else:
            skipped += 1
            continue
        background = as_list(r.get("background_description") or [])
        question = "\n".join([*map(str, background), str(r["question_text"])]).strip()
        try:
            out.append(item(question, options, gold, r.get("subject", ""), level=r.get("level")))
        except ValueError:
            skipped += 1
    if skipped:
        print(f"  m3exam {lang} {split}: skipped {skipped} questions whose answer could not be mapped")
    return out


def seaexam(config, split):
    out = []
    for r in load_dataset("SeaLLMs/SeaExam", config, split=split):
        meta = r["metadata"]
        meta = ast.literal_eval(meta) if isinstance(meta, str) else (meta or {})
        out.append(item(r["question"], as_list(r["choices"]), int(r["answer"]), meta.get("subject", "")))
    return out


def farstail(split):
    name = {"test": "Test-word.csv", "dev": "Val-word.csv"}[split]
    with urllib.request.urlopen(FARSTAIL_RAW.format(name)) as f:
        rows = list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t"))
    return [nli_item(r["premise"], r["hypothesis"], r["label"].strip(), "farstail")
            for r in rows if r["label"].strip() in NLI_LABEL]


def parsinlu(kind, split):
    """ParsiNLU entailment rows whose `category` starts with kind ("translation" =
    MNLI machine-translated into Persian, "natural" = natively written)."""
    # The Hub repository is script-based, which current `datasets` cannot load;
    # read the Hub's own parquet conversion of it instead.
    url = PARSINLU_PARQUET.format
    pick = lambda hf_split: [r for r in load_dataset("parquet", data_files={hf_split: url(hf_split)}, split=hf_split)
                             if r["category"].startswith(kind) and r["label"] in NLI_LABEL]
    rows = pick("test") if split == "test" else (pick("validation") or pick("train"))
    return [nli_item(r["sent1"], r["sent2"], r["label"], r["category"]) for r in rows]


# -------------------------------------------------------------- translated MMLU

def global_mmlu(lang, split):
    out = []
    for r in load_dataset("CohereLabs/Global-MMLU", lang, split=split):
        label = r["cultural_sensitivity_label"] if str(r["is_annotated"]).lower() == "true" else None
        out.append(item(r["question"], [r[f"option_{k}"] for k in "abcd"], LETTERS.index(r["answer"].strip()),
                        r["subject"], cs_label=label))
    return out


def mmlu_prox(lang, split):
    hf_split = {"test": "test", "dev": "validation"}[split]
    out = []
    for r in load_dataset("li-lab/MMLU-ProX", lang, split=hf_split):
        # Drop only trailing missing options: answer_index counts positions, and
        # "None" can be a real option (in Python questions, for instance).
        options = [r[f"option_{i}"] for i in range(10)]
        while options and (options[-1] is None or not str(options[-1]).strip()):
            options.pop()
        out.append(item(r["question"], ["" if o is None else o for o in options], int(r["answer_index"]),
                        r["category"]))
    return out


# ---------------------------------------------------------------------- MultiLoKo

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def multiloko_sets(repo_arg):
    """Local vs English-sourced-translated questions per language, as complete
    5-shot prompts built with MultiLoKo's own templates and few-shot examples."""
    repo = Path(repo_arg) if repo_arg else EXTERNAL / "multiloko"
    if not (repo / ".git").exists() and not (repo / "eval.py").exists():
        EXTERNAL.mkdir(exist_ok=True)
        subprocess.run(["git", "clone", MULTILOKO_REPO, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "-q", MULTILOKO_COMMIT], check=True)
    data = repo / "benchmark_data"
    if not data.exists():
        enc = repo / "benchmark_data.tar.xz.enc"
        dec = subprocess.run(["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "100000", "-d",
                              "-in", str(enc), "-pass", f"pass:{MULTILOKO_PASSWORD}"],
                             check=True, capture_output=True).stdout
        subprocess.run(["tar", "-xJ", "-C", str(repo)], input=dec, check=True)
    prompts = load_module("multiloko_prompts", repo / "examples" / "prompts.py")
    jinja_format = load_module("multiloko_utils", repo / "examples" / "utils.py").jinja_format

    def build(rows, language):
        out = []
        for r in rows:
            example = dict(r, few_shot=prompts.fewshot_examples[language][0:5])
            out.append({"id": r["id"], "prompt": jinja_format(prompts.fewshot[language], **example),
                        "targets": r["targets"]})
        return out

    def read(path):
        return [json.loads(line) for line in open(path, encoding="utf-8")]

    sets = {}
    for language in prompts.FA_TOKENS:
        if language == "english":
            continue
        sets[(language, "local")] = (language, build(read(data / language / "dev.jsonl"), language))
        for kind in ("human", "machine"):
            key = f"english_translated_{kind}_{language}"  # the key MultiLoKo's eval.py uses
            sets[(language, kind)] = (key, build(read(data / "english" / f"dev_translated_{kind}_{language}.jsonl"), language))
    return repo, sets


# ------------------------------------------------------------------------ writing

MC_TEMPLATE = """\
# One prompt for every multiple-choice set: question, lettered options, "Answer:";
# scored by the log-likelihood of each option letter; 5 in-benchmark examples.
output_type: multiple_choice
test_split: test
fewshot_split: dev
num_fewshot: 5
fewshot_config:
  sampler: default
doc_to_text: "{{question}}\\n{% for c in choices %}{{letters[loop.index0]}}. {{c}}\\n{% endfor %}Answer:"
doc_to_choice: "{{letters}}"
doc_to_target: gold
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""

GEN_TEMPLATE = """\
# MultiLoKo: the prompt already holds MultiLoKo's own 5-shot template; greedy, one line.
output_type: generate_until
test_split: test
num_fewshot: 0
doc_to_text: "{{prompt}}"
doc_to_target: "{{targets[0]}}"
generation_kwargs:
  until: ["\\n"]
  max_gen_toks: 64
  do_sample: false
  temperature: 0.0
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    ignore_punctuation: true
metadata:
  version: 1.0
"""


def stratified_sample(rows, cap, seed):
    if cap <= 0 or len(rows) <= cap:
        return rows
    rng = random.Random(seed)
    by = defaultdict(list)
    for r in rows:
        by[r["subject"]].append(r)
    keep = []
    for subject in sorted(by):
        group = by[subject]
        rng.shuffle(group)
        keep += group[:max(1, round(cap * len(group) / len(rows)))]
    rng.shuffle(keep)
    return keep[:cap]


def write_set(name, meta, test, dev, template):
    files = {}
    for split, rows in (("test", test), ("dev", dev)):
        if rows is None:
            continue
        path = DATA / f"{name}.{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": r.get("id", f"{name}/{i}"), "set": name, **meta, **r},
                                   ensure_ascii=False) + "\n")
        files[split] = str(path)
    task = f"lga_{name}"
    yaml = f"include: {template}\ntask: {task}\ndataset_path: json\ndataset_kwargs:\n  data_files:\n"
    yaml += "".join(f"    {split}: {json.dumps(path)}\n" for split, path in files.items())  # quoted for YAML
    (TASKS / f"{task}.yaml").write_text(yaml, encoding="utf-8")
    print(f"  {task:34} {len(test):6} test" + (f" / {len(dev)} dev" if dev is not None else ""))
    return task


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cap", type=int, default=0, help="max test questions per multiple-choice set (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", help="comma list of families: arabicmmlu,cmmlu,turkishmmlu,kmmlu,"
                    "m3exam,seaexam,farstail,globalmmlu,mmluprox")
    ap.add_argument("--skip-multiloko", action="store_true")
    ap.add_argument("--m3exam-zip", help="path to m3exam-data.zip if already downloaded")
    ap.add_argument("--multiloko-repo", help="path to a MultiLoKo checkout if already cloned")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    want = lambda family: only is None or family in only
    for d in (DATA, TASKS):
        d.mkdir(exist_ok=True)
    (TASKS / "_mc_template_yaml").write_text(MC_TEMPLATE, encoding="utf-8")
    (TASKS / "_gen_template_yaml").write_text(GEN_TEMPLATE, encoding="utf-8")
    mc_tasks, gen_tasks, sets, pairs = [], [], {}, []

    def mc(name, benchmark, role, lang, loader):
        test = stratified_sample(loader("test"), args.cap, args.seed)
        dev = loader("dev")
        meta = {"benchmark": benchmark, "role": role, "language": lang}
        mc_tasks.append(write_set(name, meta, test, dev, "_mc_template_yaml"))
        sets[name] = dict(meta, task=f"lga_{name}", questions=len(test))

    print("multiple-choice sets")
    gm_langs = ["ar", "zh", "tr", "ko", "it", "pt", "vi", "sw", "id"]
    if want("globalmmlu"):
        for lang in gm_langs:
            mc(f"globalmmlu_{lang}", "Global-MMLU", "translated", lang, lambda s, l=lang: global_mmlu(l, s))
    for family, name, bench, lang, loader in (
            ("arabicmmlu", "arabicmmlu_ar", "ArabicMMLU", "ar", arabicmmlu),
            ("cmmlu", "cmmlu_zh", "CMMLU", "zh", cmmlu),
            ("turkishmmlu", "turkishmmlu_tr", "TurkishMMLU", "tr", turkishmmlu),
            ("kmmlu", "kmmlu_ko", "KMMLU", "ko", kmmlu)):
        if want(family):
            mc(name, bench, "native", lang, loader)
            pairs.append({"native": name, "translated": [f"globalmmlu_{lang}"], "language": lang,
                          "pair": f"{bench} vs Global-MMLU"})
    if want("m3exam"):
        archive = m3exam_zip(args.m3exam_zip)
        for lang in M3EXAM_LANGS:
            mc(f"m3exam_{lang}", "M3Exam", "native", lang, lambda s, l=lang: m3exam(l, s, archive))
    if want("seaexam"):
        mc("seaexam_native_id", "SeaExam (native exams)", "native", "id", lambda s: seaexam("m3exam-indonesian", s))
        for lang, config in (("zh", "mmlu-chinese"), ("vi", "mmlu-vietnamese"), ("th", "mmlu-thai"), ("id", "mmlu-indonesian")):
            mc(f"seaexam_mmlu_{lang}", "SeaExam MMLU (Google-translated)", "translated", lang,
               lambda s, c=config: seaexam(c, s))
    if want("mmluprox"):
        mc("mmluprox_af", "MMLU-ProX", "translated", "af", lambda s: mmlu_prox("af", s))
    if want("m3exam"):
        translated_for = {"zh": ["globalmmlu_zh", "seaexam_mmlu_zh"], "it": ["globalmmlu_it"], "pt": ["globalmmlu_pt"],
                          "vi": ["globalmmlu_vi", "seaexam_mmlu_vi"], "th": ["seaexam_mmlu_th"], "sw": ["globalmmlu_sw"],
                          "af": ["mmluprox_af"]}
        for lang, translated in translated_for.items():
            pairs.append({"native": f"m3exam_{lang}", "translated": translated, "language": lang,
                          "pair": "M3Exam vs translated MMLU"})
    if want("seaexam"):
        pairs.append({"native": "seaexam_native_id", "translated": ["seaexam_mmlu_id", "globalmmlu_id"], "language": "id",
                      "pair": "SeaExam native vs translated MMLU (Southeast Asian stand-in for SEACrowd)"})
    if want("farstail"):
        mc("farstail_fa", "FarsTail", "native", "fa", farstail)
        mc("parsinlu_translated_fa", "ParsiNLU entailment (MNLI machine-translated)", "translated", "fa",
           lambda s: parsinlu("translation", s))
        mc("parsinlu_natural_fa", "ParsiNLU entailment (natively written)", "native", "fa", lambda s: parsinlu("natural", s))
        pairs.append({"native": "farstail_fa", "translated": ["parsinlu_translated_fa"], "language": "fa",
                      "pair": "FarsTail vs translated Persian NLI"})
        pairs.append({"native": "parsinlu_natural_fa", "translated": ["parsinlu_translated_fa"], "language": "fa",
                      "pair": "ParsiNLU natural vs translated (same dataset)"})
    pairs = [p for p in pairs if p["native"] in sets and all(t in sets for t in p["translated"])]

    multiloko = {}
    if not args.skip_multiloko:
        print("MultiLoKo (generation)")
        repo, ml_sets = multiloko_sets(args.multiloko_repo)
        for (language, kind), (key, rows) in sorted(ml_sets.items()):
            name = f"multiloko_{language}_{kind}"
            meta = {"benchmark": "MultiLoKo", "role": "native" if kind == "local" else f"translated_{kind}",
                    "language": language, "multiloko_key": key}
            gen_tasks.append(write_set(name, meta, rows, None, "_gen_template_yaml"))
            sets[name] = dict(meta, task=f"lga_{name}", questions=len(rows))
        multiloko = {"repo": str(repo), "data": str(repo / "benchmark_data"), "subset": "dev"}

    for group, tasks in (("lga_mc", mc_tasks), ("lga_multiloko", gen_tasks)):
        if tasks:
            (TASKS / f"{group}.yaml").write_text(
                f"group: {group}\ntask:\n" + "".join(f"  - {t}\n" for t in tasks)
                + "aggregate_metric_list:\n  - metric: "
                + ("acc" if group == "lga_mc" else "exact_match")
                + "\n    weight_by_size: true\nmetadata:\n  version: 1.0\n", encoding="utf-8")
    (DATA / "manifest.json").write_text(json.dumps(
        {"cap": args.cap, "seed": args.seed, "sets": sets, "pairs": pairs, "multiloko": multiloko},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{len(mc_tasks)} multiple-choice sets, {len(gen_tasks)} MultiLoKo sets, {len(pairs)} pairs;"
          f" manifest at {DATA / 'manifest.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a Claude model through the Anthropic API on the prompts lm-eval gives the
open models, and save its answers as lm-eval sample logs, so analyze.py treats it
like any other model.

Scoring differs from the open models. The API returns no token log-probabilities,
so Claude cannot be scored as they are, by the log-likelihood of each option
letter. It answers instead:
- multiple choice: a structured output restricted to the option letters, correct
  when the letter is the gold one;
- MultiLoKo: a short free-text answer, cut at the first line break as lm-eval
  cuts the open models' answers, and scored later by MultiLoKo's eval.py.
Each prompt is the 5-shot prompt lm-eval builds for the open models, with the same
five examples (lm-eval builds it here, with run.sh's seeds), sent as the user
message after a one-line system instruction. Thinking is off, so Claude answers
directly, as the open models do.

Requests go through the Message Batches API, at half the standard price. Before
the first batch, one request of each kind is sent directly, so that a request the
API rejects stops the run before it costs anything; results are written only if
no request was rejected as invalid. The run can be resumed: batch ids and answers are kept under results/claude/<model>/, and
running the script again collects finished batches and resubmits only what is
missing or failed. Answers are keyed by a hash of their request, so a changed
prompt is asked again rather than reused.

Usage (needs `pip install anthropic` and ANTHROPIC_API_KEY):
  python run_claude.py --limit 5     # smoke test: 5 questions per set, about a cent
  python run_claude.py               # everything: shows the estimated cost, asks first
  python run_claude.py --yes         # the same without asking (for nohup)
  python run_claude.py --status      # progress of submitted batches; submits nothing
"""
import argparse
import datetime
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GROUPS = ("lga_mc", "lga_multiloko")
# $ per million input / output tokens on the standard API (checked 2026-10-25);
# batches cost half. Used only for the estimate and the final tally.
PRICES = {"claude-haiku-5-5": (0.10, 0.50)}
MAX_REQUESTS, MAX_BYTES = 100_000, 200 * 2 ** 20  # per batch; the API allows 100,000 requests or 256 MB
INSTRUCTION = {
    "multiple_choice": "The message gives example questions, each followed by its answer, and then one more "
                       "question. Answer that last question with the letter of the correct option.",
    "generate_until": "The message gives example questions, each followed by its answer, and then one more "
                      "question. Answer that last question as the examples are answered: only the short answer, "
                      "in the same language, on one line.",
}


def directory(model):
    return re.sub(r"[\"<>:/|\\?*\[\]]+", "__", model)  # as lm-eval names a model's results directory


# ------------------------------------------------------------------------ prompts

def build_task(name, limit):
    """One task's questions with the prompts lm-eval builds, seeded as run.sh's
    lm_eval runs are (--seed 0,1234,1234,1234). Each task's few-shot sampling has
    its own generator, so tasks can be built separately and in any order."""
    from lm_eval.tasks import TaskManager
    random.seed(0)
    np.random.seed(1234)
    task = TaskManager(include_path=str(HERE / "tasks")).load([name])["tasks"][name]
    task.set_fewshot_seed(seed=1234)
    task.build_all_requests(limit=limit, rank=0, world_size=1)
    first = {}
    for instance in task.instances:  # multiple choice has one instance per option, all with the same prompt
        first.setdefault(instance.doc_id, instance)
    return [(name, task.get_config("output_type"), doc_id, inst.doc, inst.arguments[0])
            for doc_id, inst in sorted(first.items())]


def build_items(groups, limit, workers, cache):
    """[(group, task, output_type, doc_id, doc, prompt)] for every question in the
    groups, cached in `cache` until tasks/ or data/ change."""
    stamp = sorted((str(p.relative_to(HERE)), p.stat().st_size, p.stat().st_mtime_ns)
                   for d in ("tasks", "data") for p in (HERE / d).iterdir())
    key = hashlib.sha256(json.dumps([groups, limit, stamp]).encode()).hexdigest()
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            if json.loads(f.readline()) == key:
                return [tuple(json.loads(line)) for line in f]
    names = []
    for group in groups:
        listed = re.findall(r"^  - (\S+)$", (HERE / "tasks" / f"{group}.yaml").read_text(encoding="utf-8"), re.M)
        names += [(group, name) for name in listed]
    print(f"building {len(names)} sets of prompts with lm-eval ({workers} processes)")
    items = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (group, _), built in zip(names, pool.map(build_task, [n for _, n in names], [limit] * len(names))):
            items += [(group, *item) for item in built]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt", encoding="utf-8") as f:
        f.write(json.dumps(key) + "\n")
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    return items


def request(model, output_type, doc, prompt):
    params = {"model": model, "max_tokens": 64, "system": INSTRUCTION[output_type],
              "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]}
    if output_type == "multiple_choice":
        params["max_tokens"] = 32
        params["output_config"] = {"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"answer": {"type": "string", "enum": list(doc["letters"])}},
            "required": ["answer"], "additionalProperties": False}}}
    return params


def custom_id(task, doc_id, params):
    """Unique per request and changes with it; [A-Za-z0-9_-], at most 64 characters."""
    digest = hashlib.sha256(json.dumps(params, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:10]
    return f"{task.removeprefix('lga_')}-{doc_id}-{digest}"


# ------------------------------------------------------------------------ batches

def load_answers(path):
    """{custom_id: record}: the latest result of each request."""
    answers = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            record = json.loads(line)
            answers[record["custom_id"]] = record
    return answers


def final(record):
    """A result to keep: a reply (refusals included), or a request the API rejects as invalid."""
    return record["result"] == "succeeded" or record.get("error") == "invalid_request_error"


def collect(client, batches, answers_path):
    """Saves the results of every batch that has ended; returns the batches still running."""
    running = []
    for b in batches:
        if b["collected"]:
            continue
        batch = client.messages.batches.retrieve(b["id"])
        if batch.processing_status != "ended":
            running.append((b, batch))
            continue
        with answers_path.open("a", encoding="utf-8") as f:
            for r in client.messages.batches.results(b["id"]):
                record = {"custom_id": r.custom_id, "result": r.result.type}
                if r.result.type == "succeeded":
                    message = r.result.message
                    record.update(text="".join(c.text for c in message.content if c.type == "text"),
                                  stop_reason=message.stop_reason, input_tokens=message.usage.input_tokens,
                                  output_tokens=message.usage.output_tokens)
                elif r.result.type == "errored":
                    error = getattr(r.result.error, "error", r.result.error)
                    record.update(error=getattr(error, "type", "error"), message=getattr(error, "message", ""))
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        b["collected"] = True
    return running


def chunks(todo):
    """Splits (custom_id, params) pairs into batches within the API's limits."""
    chunk, size = [], 0
    for cid, params in todo:
        n = len(json.dumps(params, ensure_ascii=False).encode()) + 200
        if chunk and (len(chunk) == MAX_REQUESTS or size + n > MAX_BYTES):
            yield chunk
            chunk, size = [], 0
        chunk.append((cid, params))
        size += n
    if chunk:
        yield chunk


def estimate(client, todo):
    """Input and output tokens of the requests, from exact counts on a sample of 40."""
    sample = random.Random(0).sample(todo, min(40, len(todo)))
    chars = lambda p: len(p["system"]) + len(p["messages"][0]["content"])
    counted = sum(client.messages.count_tokens(**{k: v for k, v in p.items() if k != "max_tokens"}).input_tokens
                  for _, p in sample)
    tokens_in = counted / sum(chars(p) for _, p in sample) * sum(chars(p) for _, p in todo)
    tokens_out = sum(10 if "output_config" in p else 20 for _, p in todo)  # a JSON letter; a short answer
    return tokens_in, tokens_out


def preflight(client, todo):
    """Sends one request of each kind directly, so that a request the API rejects
    fails here, for a fraction of a cent, rather than across a whole batch."""
    kinds = {}
    for cid, params in todo:
        kinds.setdefault("output_config" in params, (cid, params))
    for cid, params in kinds.values():
        message = client.messages.create(**params)
        text = "".join(c.text for c in message.content if c.type == "text")
        print(f"check {cid}: {message.stop_reason}, {text[:60]!r}")


def dollars(model, tokens_in, tokens_out):
    price_in, price_out = PRICES.get(model, (float("nan"), float("nan")))
    return (tokens_in * price_in + tokens_out * price_out) / 1e6 / 2  # batches are half price


# ------------------------------------------------------------------------ results

def short_answer(text, prompt):
    """The reply cut at its first line break, as lm-eval cuts the open models'
    answers, without a repeat of the prompt's closing answer label ("Svar:")."""
    label = prompt.rsplit("\n", 1)[-1].strip()
    answer = text.strip().split("\n")[0].strip()
    return answer[len(label):].strip() if label and answer.startswith(label) else answer


def write_results(model, items, answers, limit):
    """Writes lm-eval-style samples_<task>_<stamp>.jsonl and results_<stamp>.json files."""
    stamp = datetime.datetime.now().isoformat().replace(":", "-")
    per_task, counts = defaultdict(list), Counter()
    for group, task, output_type, doc_id, doc, prompt, cid, _ in items:
        record = answers[cid]
        text = record.get("text", "")
        counts["refused"] += record.get("stop_reason") == "refusal"
        if output_type == "multiple_choice":
            try:
                letter = json.loads(text)["answer"]
            except (ValueError, KeyError, TypeError):
                letter = ""
                counts["unparsed"] += bool(text)  # an empty reply is a refusal, counted above
            gold = doc["letters"][doc["gold"]]
            sample = {"doc_id": doc_id, "doc": doc, "target": gold, "resps": [[text]], "filtered_resps": [letter],
                      "acc": float(letter == gold)}
        else:
            sample = {"doc_id": doc_id, "doc": doc, "target": doc["targets"], "resps": [[text]],
                      "filtered_resps": [short_answer(text, prompt)]}
        per_task[(group, task)].append(sample)
    for (group, task), samples in per_task.items():
        out = HERE / "results" / group / directory(model)
        out.mkdir(parents=True, exist_ok=True)
        with (out / f"samples_{task}_{stamp}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
    for group in sorted({g for g, _ in per_task}):
        results = {}
        for (g, task), samples in sorted(per_task.items()):
            if g == group:
                results[task] = {"acc,none": float(np.mean([s["acc"] for s in samples]))} if "acc" in samples[0] \
                    else {"answered": len(samples)}
        config = {"model": "anthropic-message-batches", "model_args": f"model={model}", "limit": limit,
                  "scoring": "generated"}
        (HERE / "results" / group / directory(model) / f"results_{stamp}.json").write_text(
            json.dumps({"config": config, "results": results}, indent=2) + "\n", encoding="utf-8")
    return counts


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--limit", type=int, help="questions per set, for a smoke test (like lm-eval's --limit)")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--yes", action="store_true", help="submit without asking")
    ap.add_argument("--status", action="store_true", help="report progress only; submits nothing")
    ap.add_argument("--no-wait", action="store_true", help="submit and exit; run again later to collect")
    ap.add_argument("--poll", type=int, default=60, help="seconds between checks while batches run")
    ap.add_argument("--retries", type=int, default=3, help="rounds of resubmitting failed requests")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1), help="processes building prompts")
    args = ap.parse_args()
    run(args, lambda: __import__("anthropic").Anthropic())


def run(args, make_client):
    root = HERE / "results" / "claude" / directory(args.model)
    root.mkdir(parents=True, exist_ok=True)
    batches_path, answers_path = root / "batches.json", root / "answers.jsonl"
    batches = json.loads(batches_path.read_text(encoding="utf-8")) if batches_path.exists() else []
    save = lambda: batches_path.write_text(json.dumps(batches) + "\n", encoding="utf-8")

    if args.status:
        client = make_client()
        running = collect(client, batches, answers_path)
        save()
        for b, batch in running:
            c = batch.request_counts
            print(f"{b['id']}: {batch.processing_status}, {c.processing} processing, {c.succeeded} succeeded, "
                  f"{c.errored} errored")
        answers = load_answers(answers_path)
        print(f"{len(running)} batches running; {sum(map(final, answers.values()))} answers saved")
        return

    if not (HERE / "tasks" / "lga_mc.yaml").exists():
        sys.exit("tasks/ is missing: run python prepare.py first")
    groups = args.groups.split(",")
    cache = root / f"prompts-{args.limit or 'all'}.jsonl.gz"
    items = []
    for group, task, output_type, doc_id, doc, prompt in build_items(groups, args.limit, args.workers, cache):
        params = request(args.model, output_type, doc, prompt)
        items.append((group, task, output_type, doc_id, doc, prompt, custom_id(task, doc_id, params), params))
    print(f"{len(items)} questions in {len({i[1] for i in items})} sets")

    client = make_client()  # after the prompt-building processes have forked
    for round_ in range(args.retries + 1):
        running = collect(client, batches, answers_path)
        save()
        answers = load_answers(answers_path)
        in_flight = {cid for b, _ in running for cid in b["custom_ids"]}
        todo = [(i[6], i[7]) for i in items if i[6] not in in_flight and not (i[6] in answers and final(answers[i[6]]))]
        if todo and round_ < args.retries:
            tokens_in, tokens_out = estimate(client, todo)
            print(f"{len(todo)} requests to submit: about {tokens_in / 1e6:.1f}M input and {tokens_out / 1e6:.1f}M "
                  f"output tokens, about ${dollars(args.model, tokens_in, tokens_out):.2f} at batch prices")
            if round_ == 0:
                if not args.yes:
                    try:
                        go = input("Submit? [y/N] ").strip().lower() == "y"
                    except EOFError:
                        go = False
                    if not go:
                        sys.exit("nothing submitted (use --yes to submit without asking)")
                preflight(client, todo)
            for chunk in chunks(todo):
                batch = client.messages.batches.create(requests=[{"custom_id": c, "params": p} for c, p in chunk])
                batches.append({"id": batch.id, "custom_ids": [c for c, _ in chunk], "collected": False,
                                "created": datetime.datetime.now().isoformat(timespec="seconds")})
                save()
                print(f"submitted batch {batch.id} ({len(chunk)} requests)")
            running = collect(client, batches, answers_path)
            save()
        if not running:
            break
        if args.no_wait:
            sys.exit("batches are running; run this command again later to collect the answers")
        while running:
            done = sum(batch.request_counts.succeeded + batch.request_counts.errored for _, batch in running)
            print(f"{time.strftime('%H:%M:%S')} waiting for {len(running)} batches ({done} requests done)")
            time.sleep(args.poll)
            running = collect(client, batches, answers_path)
            save()

    answers = load_answers(answers_path)
    missing = [i for i in items if i[6] not in answers or not final(answers[i[6]])]
    if missing:
        sys.exit(f"{len(missing)} requests still have no answer after {args.retries} rounds; run again to retry them")
    invalid = [answers[i[6]] for i in items if answers[i[6]]["result"] != "succeeded"]
    if invalid:  # a fault in the requests, not an answer: write nothing rather than score them as wrong
        sys.exit(f"the API rejected {len(invalid)} requests as invalid, e.g. {invalid[0].get('message')!r}; "
                 "no results written")
    counts = write_results(args.model, items, answers, args.limit)
    used = [answers[i[6]] for i in items]
    tokens_in, tokens_out = sum(r.get("input_tokens", 0) for r in used), sum(r.get("output_tokens", 0) for r in used)
    print(f"wrote results/<group>/{directory(args.model)}/ for {len(items)} questions: {counts['refused']} refused, "
          f"{counts['unparsed']} unparsed answers; "
          f"{tokens_in / 1e6:.2f}M input and {tokens_out / 1e6:.2f}M output tokens, about "
          f"${dollars(args.model, tokens_in, tokens_out):.2f} at batch prices")


if __name__ == "__main__":
    main()

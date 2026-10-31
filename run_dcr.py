#!/usr/bin/env python3
"""
run_dcr.py - Run the DCR model stages through the Anthropic API, with full logging.

Stages (each depends on the one before):
  read     every reader (real and sham) x document x repetition, per prompts/experiment.yaml
  extract  one claim-extraction call per successful reading
  edit     editors over the extracted claims, per corpus condition (main runs + editor noise floor)

Commands:
  python run_dcr.py pilot               small direct run of all three stages (~15 calls) to check prompts
  python run_dcr.py submit read         submit a batch; requests that already succeeded are skipped
  python run_dcr.py status              progress of submitted batches
  python run_dcr.py collect read        download finished results into runs/outputs/read.jsonl
  python run_dcr.py submit extract      ... then collect extract, submit edit, collect edit
  python run_dcr.py summary             counts, failures, parse errors, token usage per stage

Re-running "submit <stage>" after "collect" resubmits only what is missing or failed.

Readers are blind to authors and titles: they see a document ID, its year, and its passages.

Logging: runs/requests/<stage>.jsonl records each request's parameters and prompt hashes (not the prompt
text, which contains copyrighted passages). runs/outputs/<stage>.jsonl records every model response,
its usage, and the parsed JSON. runs/batches.json records batch IDs. Each record carries the git commit.

Requires: pip install anthropic pyyaml   and the ANTHROPIC_API_KEY environment variable.

Full run through Claude Code (parallel, resumable; rerun the same command to finish anything missed):
  python run_dcr.py run all --backend claude-code --workers 4
or one stage at a time:  run read | run extract | run edit   (outputs go to runs_cc/)

Smoke test without an API key:
  python run_dcr.py pilot --backend claude-code
runs the pilot through the local Claude Code CLI ("claude -p") in an empty temporary folder. This is
for checking that prompts and parsing work. Claude Code does not let us set temperature and may add
its own instructions, so its outputs are labeled backend=claude-code and are not used for results.
"""

import argparse
import csv
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict
from pathlib import Path

import yaml

PROMPTS = Path("prompts")
CORPUS = Path("corpus")
READER_CORPORA = ["legitimacy", "tce", "splice"]  # document sets that readers read
CID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def git_state():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "prompts"], capture_output=True,
                               text=True).stdout.strip()
        return commit or "unknown", bool(dirty)
    except FileNotFoundError:
        return "unknown", False


def load_yaml(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


def fill(s, **kw):
    for k, v in kw.items():
        s = s.replace(f"<<{k}>>", v)
    return s


class Project:
    def __init__(self, outdir, experiment=None):
        self.exp_path = Path(experiment) if experiment else PROMPTS / "experiment.yaml"
        self.exp = load_yaml(self.exp_path)
        self.reader_tpl = load_yaml(PROMPTS / "readers" / "_template.yaml")
        self.extract_tpl = load_yaml(PROMPTS / "claim_extraction.yaml")
        self.editor_tpl = load_yaml(PROMPTS / "editors" / "_template.yaml")
        names = self.exp["readers"]["real"] + self.exp["readers"]["sham"]
        self.readers = {n: load_yaml(PROMPTS / "readers" / f"{n}.yaml") for n in names}
        self.editors = {n: load_yaml(PROMPTS / "editors" / f"{n}.yaml") for n in self.exp["editors"]}

        # documents: corpus membership and year (no titles or authors are shown to models)
        self.docs, self.doc_order = {}, defaultdict(list)
        for c in READER_CORPORA:
            with open(CORPUS / f"{c}_selected.csv", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.docs[r["openalex_id"]] = {"corpus": c, "year": r["year"]}
                    self.doc_order[c].append(r["openalex_id"])
        self.passages = defaultdict(list)
        with open(CORPUS / "passages.jsonl", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                self.passages[p["openalex_id"]].append(p)

        cor = self.exp["corpora"]
        self.construct_for = {"legitimacy": cor["legitimacy"]["construct"], "tce": cor["tce"]["construct"],
                              "splice": cor["tce_spliced"]["construct"]}
        self.conditions = {
            "legitimacy": self.doc_order["legitimacy"],
            "tce": self.doc_order["tce"],
            "tce_spliced": self.doc_order["tce"] + self.doc_order["splice"],
        }
        self.condition_construct = {k: cor[k]["construct"] for k in self.conditions}

        self.out = Path(outdir)
        for sub in ("requests", "outputs"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.commit, self.dirty = git_state()

    # -- prompt construction --------------------------------------------------

    def model_params(self, role):
        m = self.exp["models"][role]
        return {"model": m["model"], "temperature": m["temperature"], "max_tokens": m["max_tokens"]}

    def reader_request(self, persona, oid):
        p = self.readers[persona]
        construct = self.construct_for[self.docs[oid]["corpus"]]
        passages = "\n\n".join(
            f"[{x['passage_id']}]" + (f" (page {x['page']})" if x["page"] else "") + f"\n{x['text']}"
            for x in self.passages[oid])
        document = f"ID: {oid}\nYear: {self.docs[oid]['year']}"
        user = fill(self.reader_tpl["user"], persona_block=p["persona_block"], output_schema=p["output_schema"])
        user = fill(user, construct=construct, document=document, passages=passages)
        system = fill(self.reader_tpl["system"], construct=construct)
        return system, user

    def extract_request(self, oid, reader_text):
        construct = self.construct_for[self.docs[oid]["corpus"]]
        document = f"ID: {oid}\nYear: {self.docs[oid]['year']}"
        user = fill(self.extract_tpl["user"], construct=construct, document=document, reader_output=reader_text)
        return self.extract_tpl["system"], user

    def editor_request(self, editor, condition, rep, reads, extracts):
        construct = self.condition_construct[condition]
        docs = self.conditions[condition]
        documents = "\n".join(f"{oid} ({self.docs[oid]['year']})" for oid in docs)
        blocks = []
        for oid in docs:
            lines = [f"=== DOCUMENT {oid} ({self.docs[oid]['year']})"]
            for persona in self.exp["readers"]["real"]:
                lines.append(f"--- reader: {persona}")
                ex = extracts.get((persona, oid, rep))
                rd = reads.get((persona, oid, rep))
                if not ex or not ex.get("parsed"):
                    lines.append("(no reading available)")
                    continue
                st = ex["parsed"].get("stance", {})
                lines.append(f"Stance: {st.get('label', '?')} (target: {st.get('target', '?')}). {st.get('text', '')}")
                lines.append("Characterization:")
                for k in ex["parsed"].get("characterization", []):
                    lines.append(f"- {k.get('id', '')}: {k.get('text', '')}")
                summary = (rd or {}).get("parsed", {}) or {}
                if summary.get("reading"):
                    lines.append(f"Reader's summary: {summary['reading']}")
            blocks.append("\n".join(lines))
        user = fill(self.editor_tpl["user"], editor_role=self.editors[editor]["editor_role"])
        user = fill(user, construct=construct, documents=documents, readings="\n\n".join(blocks))
        system = fill(self.editor_tpl["system"], construct=construct)
        return system, user

    # -- task lists -----------------------------------------------------------

    def read_tasks(self, pilot=False):
        tasks = []
        runs = self.exp["runs"]["reading"]
        for kind in ("real", "sham"):
            personas = self.exp["readers"][kind]
            corpora, reps = runs[kind]["documents"], runs[kind]["repetitions"]
            if pilot:
                personas = personas if kind == "real" else personas[:1]
                corpora, reps = ["legitimacy"], 1
            for persona in personas:
                for c in corpora:
                    docs = self.doc_order[c][: (2 if kind == "real" else 1)] if pilot else self.doc_order[c]
                    for oid in docs:
                        for r in range(1, reps + 1):
                            cid = f"read__{persona}__{oid}__r{r}"
                            system, user = self.reader_request(persona, oid)
                            tasks.append((cid, "readers", system, user))
        return tasks

    def extract_tasks(self, reads):
        tasks = []
        for (persona, oid, rep), rec in sorted(reads.items()):
            cid = f"extract__{persona}__{oid}__r{rep}"
            reader_text = json.dumps(rec["parsed"], indent=1, ensure_ascii=False) if rec.get("parsed") else rec["text"]
            system, user = self.extract_request(oid, reader_text)
            tasks.append((cid, "extraction", system, user))
        return tasks

    def edit_tasks(self, reads, extracts, pilot=False):
        tasks = []
        if pilot:
            pilot_docs = self.doc_order["legitimacy"][:2]
            saved = self.conditions["legitimacy"]
            self.conditions["legitimacy"] = pilot_docs
            system, user = self.editor_request("synthesizing", "legitimacy", 1, reads, extracts)
            self.conditions["legitimacy"] = saved
            return [("edit__synthesizing__legitimacy__r1", "editors", system, user)]
        main = self.exp["runs"]["editing"]["main"]
        for c in main["corpora"]:
            for e in main["editors"]:
                for r in main["reader_repetitions"]:
                    system, user = self.editor_request(e, c, r, reads, extracts)
                    tasks.append((f"edit__{e}__{c}__r{r}", "editors", system, user))
        fl = self.exp["runs"]["editing"]["editor_noise_floor"]
        for c in fl["corpora"]:
            for e in fl["editors"]:
                r = fl["reader_repetition"]
                system, user = self.editor_request(e, c, r, reads, extracts)
                for k in range(1, fl["repetitions"] + 1):
                    tasks.append((f"edit__{e}__{c}__r{r}__f{k}", "editors", system, user))
        return tasks


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


def parse_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            try:
                return json.loads(t[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None


def load_outputs(proj, stage):
    """Latest successful record per custom_id, keyed by parsed ID parts."""
    path = proj.out / "outputs" / f"{stage}.jsonl"
    best = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            if rec["status"] != "succeeded":
                continue
            prev = best.get(rec["custom_id"])
            # Prefer a record that parsed; among equals, the latest.
            if prev is None or rec.get("parse_ok") or not prev.get("parse_ok"):
                best[rec["custom_id"]] = rec
    keyed = {}
    for cid, rec in best.items():
        parts = cid.split("__")
        if stage in ("read", "extract"):
            keyed[(parts[1], parts[2], int(parts[3][1:]))] = rec
        else:
            keyed[cid] = rec
    return best, keyed


_WRITE_LOCK = threading.Lock()


def write_output(proj, stage, cid, status, message=None, error=None):
    rec = {"custom_id": cid, "status": status, "collected_utc": now(), "git_commit": proj.commit}
    if message is not None:
        text = "".join(getattr(b, "text", "") for b in message.content)
        rec.update({"model": message.model, "stop_reason": message.stop_reason,
                    "usage": {"input_tokens": message.usage.input_tokens,
                              "output_tokens": message.usage.output_tokens},
                    "text": text})
        parsed = parse_json(text)
        rec["parsed"], rec["parse_ok"] = parsed, parsed is not None
    if error is not None:
        rec["error"] = str(error)
    with _WRITE_LOCK, open(proj.out / "outputs" / f"{stage}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def log_requests(proj, stage, tasks):
    with open(proj.out / "requests" / f"{stage}.jsonl", "a", encoding="utf-8") as f:
        for cid, role, system, user in tasks:
            f.write(json.dumps({"custom_id": cid, "logged_utc": now(), "git_commit": proj.commit,
                                "prompts_dirty": proj.dirty, "experiment": str(proj.exp_path),
                                **proj.model_params(role),
                                "system_sha256": sha(system), "user_sha256": sha(user),
                                "user_chars": len(user)}) + "\n")


def params_for(proj, role, system, user):
    return {**proj.model_params(role), "system": system, "messages": [{"role": "user", "content": user}]}


def get_client():
    import anthropic
    return anthropic.Anthropic()


class ClaudeCodeClient:
    """Minimal stand-in for anthropic.Anthropic().messages.create, using `claude -p`."""

    def __init__(self, binary=None):
        found = shutil.which(binary) if binary else shutil.which("claude")
        if binary and not found and Path(binary).exists():
            found = binary
        self.binary = str(Path(found).resolve()) if found else None  # absolute: we run from a temp folder
        if not self.binary:
            sys.exit("Could not find the Claude Code CLI ('claude') on PATH. Pass --claude-bin <path>.")
        self.workdir = tempfile.mkdtemp(prefix="dcr_cc_")  # empty folder: nothing for tools to read
        self.messages = types.SimpleNamespace(create=self.create)
        self.full_flags = True

    def _run(self, system, user, model):
        if self.full_flags:
            cmd = [self.binary, "-p", "--output-format", "json", "--model", model,
                   "--max-turns", "1", "--system-prompt", system]
            stdin = user
        else:  # fallback if this CLI version rejects a flag: fold the system prompt into the message
            cmd = [self.binary, "-p", "--output-format", "json"]
            stdin = f"{system}\n\n{user}"
        return subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
                               cwd=self.workdir, timeout=900)

    def create(self, model, system, messages, **_ignored):  # temperature/max_tokens not settable here
        user = messages[0]["content"]
        res = self._run(system, user, model)
        if res.returncode != 0 and self.full_flags and ("unknown option" in res.stderr.lower()
                                                        or "error: option" in res.stderr.lower()):
            print("  (CLI rejected a flag; retrying with system prompt folded into the message)")
            self.full_flags = False
            res = self._run(system, user, model)
        if res.returncode != 0:
            raise RuntimeError(f"claude -p failed ({res.returncode}): {res.stderr.strip()[:300]}")
        try:
            out = json.loads(res.stdout)
        except json.JSONDecodeError:
            out = {"result": res.stdout}
        # Claude Code reports rate limits and API errors inside a "successful" JSON envelope.
        # Treat those as failures so they are retried rather than saved as readings.
        if out.get("is_error") or out.get("subtype") not in (None, "success") or out.get("api_error_status"):
            raise RuntimeError(f"Claude Code reported an error: subtype={out.get('subtype')} "
                               f"api_error_status={out.get('api_error_status')} result={str(out.get('result'))[:200]}")
        u = out.get("usage") or {}
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(text=out.get("result", ""))],
            model=f"claude-code:{model}", stop_reason=out.get("subtype", "unknown"),
            usage=types.SimpleNamespace(input_tokens=u.get("input_tokens", 0) or 0,
                                        output_tokens=u.get("output_tokens", 0) or 0))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def tasks_for(proj, stage, pilot=False):
    if stage == "read":
        return proj.read_tasks(pilot)
    _, reads = load_outputs(proj, "read")
    if stage == "extract":
        if not reads:
            sys.exit("No collected readings yet. Run and collect the read stage first.")
        return proj.extract_tasks(reads)
    _, extracts = load_outputs(proj, "extract")
    if not proj.exp["runs"]["editing"]["main"]["corpora"] and not pilot:
        return []  # this experiment configures no editor runs
    if not extracts:
        sys.exit("No collected extractions yet. Run and collect the extract stage first.")
    return proj.edit_tasks(reads, extracts, pilot)


def warn_dirty(proj):
    if proj.dirty:
        print("WARNING: prompts/ has uncommitted changes. Commit (or log in CHANGELOG.md) before real runs.")


def cmd_submit(proj, stage, client):
    warn_dirty(proj)
    done, _ = load_outputs(proj, stage)
    tasks = [t for t in tasks_for(proj, stage) if t[0] not in done]
    if not tasks:
        print(f"Nothing to submit for {stage}: all requests already succeeded.")
        return
    bad = [t[0] for t in tasks if not CID_RE.match(t[0])]
    if bad:
        sys.exit(f"Invalid custom_id(s): {bad[:3]}")
    log_requests(proj, stage, tasks)
    batch = client.messages.batches.create(requests=[
        {"custom_id": cid, "params": params_for(proj, role, system, user)} for cid, role, system, user in tasks])
    path = proj.out / "batches.json"
    batches = json.loads(path.read_text()) if path.exists() else []
    batches.append({"stage": stage, "batch_id": batch.id, "submitted_utc": now(), "n_requests": len(tasks),
                    "git_commit": proj.commit, "collected": False})
    path.write_text(json.dumps(batches, indent=2))
    print(f"Submitted {len(tasks)} {stage} requests as batch {batch.id}. Check with: python run_dcr.py status")


def cmd_status(proj, client):
    path = proj.out / "batches.json"
    if not path.exists():
        print("No batches submitted yet.")
        return
    for b in json.loads(path.read_text()):
        if b["collected"]:
            print(f"{b['stage']:<8} {b['batch_id']}  collected ({b['n_requests']} requests)")
            continue
        info = client.messages.batches.retrieve(b["batch_id"])
        c = info.request_counts
        print(f"{b['stage']:<8} {b['batch_id']}  {info.processing_status}: processing {c.processing}, "
              f"succeeded {c.succeeded}, errored {c.errored}, expired {c.expired}, canceled {c.canceled}")


def cmd_collect(proj, stage, client):
    path = proj.out / "batches.json"
    batches = json.loads(path.read_text()) if path.exists() else []
    pending = [b for b in batches if b["stage"] == stage and not b["collected"]]
    if not pending:
        print(f"No uncollected {stage} batches.")
        return
    for b in pending:
        info = client.messages.batches.retrieve(b["batch_id"])
        if info.processing_status != "ended":
            print(f"Batch {b['batch_id']} still {info.processing_status}; try again later.")
            continue
        n = defaultdict(int)
        for res in client.messages.batches.results(b["batch_id"]):
            kind = res.result.type
            if kind == "succeeded":
                rec = write_output(proj, stage, res.custom_id, "succeeded", message=res.result.message)
                n["parse_fail"] += not rec["parse_ok"]
            else:
                write_output(proj, stage, res.custom_id, kind, error=getattr(res.result, "error", kind))
            n[kind] += 1
        b["collected"], b["collected_utc"] = True, now()
        print(f"Collected batch {b['batch_id']}: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))
    path.write_text(json.dumps(batches, indent=2))
    print(f"If anything failed, rerun: python run_dcr.py submit {stage}")


def run_direct(proj, stage, tasks, client):
    log_requests(proj, stage, tasks)
    for cid, role, system, user in tasks:
        for attempt in range(4):
            try:
                msg = client.messages.create(**params_for(proj, role, system, user))
                rec = write_output(proj, stage, cid, "succeeded", message=msg)
                print(f"  {cid}: ok" + ("" if rec["parse_ok"] else "  (JSON PARSE FAILED)"))
                break
            except Exception as e:  # rate limits, overload: back off and retry
                if attempt == 3:
                    write_output(proj, stage, cid, "errored", error=e)
                    print(f"  {cid}: ERROR {e}")
                else:
                    time.sleep(5 * 2 ** attempt)


def call_with_retry(proj, stage, client, task):
    cid, role, system, user = task
    for attempt in range(4):
        try:
            msg = client.messages.create(**params_for(proj, role, system, user))
            rec = write_output(proj, stage, cid, "succeeded", message=msg)
            return cid, "ok" if rec["parse_ok"] else "ok (JSON PARSE FAILED)"
        except Exception as e:  # rate limits, overload, CLI hiccups: back off and retry
            if attempt == 3:
                write_output(proj, stage, cid, "errored", error=e)
                return cid, f"ERROR {str(e)[:150]}"
            time.sleep(10 * 2 ** attempt)


def run_parallel(proj, stage, client, workers):
    done, _ = load_outputs(proj, stage)
    tasks = [t for t in tasks_for(proj, stage) if t[0] not in done]
    total = len(tasks)
    print(f"\n{stage}: {total} to run ({len(done)} already done), {workers} in parallel")
    if not total:
        return 0
    log_requests(proj, stage, tasks)
    start, finished, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call_with_retry, proj, stage, client, t) for t in tasks]
        for fut in as_completed(futures):
            cid, status = fut.result()
            finished += 1
            failed += status.startswith("ERROR")
            rate = (time.time() - start) / finished
            eta = rate * (total - finished) / 60
            print(f"  [{finished}/{total}] {cid}: {status}   (~{eta:.0f} min left)")
    print(f"{stage} done: {total - failed} succeeded, {failed} failed"
          + ("  -> rerun the same command to retry failures" if failed else ""))
    return failed


def cmd_run(proj, stage, client, workers):
    warn_dirty(proj)
    stages = ["read", "extract", "edit"] if stage == "all" else [stage]
    for st in stages:
        failed = run_parallel(proj, st, client, workers)
        if failed and st != stages[-1]:
            print(f"\nStopping before the next stage because {failed} {st} call(s) failed. "
                  f"Rerun the same command; finished calls are skipped.")
            return
    cmd_summary(proj)


def cmd_pilot(client, outdir="runs_pilot"):
    proj = Project(outdir)
    warn_dirty(proj)
    print(f"Pilot output in {proj.out}/ (separate from real runs)\n")
    for stage in ("read", "extract", "edit"):
        done, _ = load_outputs(proj, stage)
        tasks = [t for t in tasks_for(proj, stage, pilot=True) if t[0] not in done]
        print(f"{stage}: {len(tasks)} calls")
        run_direct(proj, stage, tasks, client)
    _, edits = load_outputs(proj, "edit")
    for cid, rec in edits.items():
        ov = (rec.get("parsed") or {}).get("overall", {})
        print(f"\nPilot editor judgment: {ov.get('judgment')} ({ov.get('confidence')}): {ov.get('rationale', '')[:300]}")
    cmd_summary(proj)


def cmd_summary(proj):
    print("\nStage     records  succeeded  parse_fail  input_tok  output_tok")
    for stage in ("read", "extract", "edit"):
        path = proj.out / "outputs" / f"{stage}.jsonl"
        if not path.exists():
            continue
        recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        ok = [r for r in recs if r["status"] == "succeeded"]
        pf = sum(not r.get("parse_ok") for r in ok)
        tin = sum(r["usage"]["input_tokens"] for r in ok)
        tout = sum(r["usage"]["output_tokens"] for r in ok)
        trunc = sum(r.get("stop_reason") == "max_tokens" for r in ok)
        print(f"{stage:<9} {len(recs):>7}  {len(ok):>9}  {pf:>10}  {tin:>9}  {tout:>10}"
              + (f"   ({trunc} hit max_tokens)" if trunc else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["pilot", "run", "submit", "status", "collect", "summary"])
    ap.add_argument("stage", nargs="?", choices=["read", "extract", "edit", "all"])
    ap.add_argument("--workers", type=int, default=4, help="parallel calls for 'run' (default 4)")
    ap.add_argument("--outdir", default=None, help="default: runs (api) or runs_cc (claude-code)")
    ap.add_argument("--backend", choices=["api", "claude-code"], default="api",
                    help="claude-code: run the pilot through the local Claude Code CLI (smoke test only)")
    ap.add_argument("--claude-bin", default=None, help="Path to the claude CLI if not on PATH")
    ap.add_argument("--experiment", default=None,
                    help="experiment config (default prompts/experiment.yaml), e.g. a sensitivity check")
    args = ap.parse_args()
    if args.command in ("submit", "collect", "run") and not args.stage:
        sys.exit(f"Usage: python run_dcr.py {args.command} read|extract|edit" + ("|all" if args.command == "run" else ""))
    if args.stage == "all" and args.command != "run":
        sys.exit("'all' works only with the run command.")
    outdir = args.outdir or ("runs_cc" if args.backend == "claude-code" else "runs")

    if args.command == "summary":
        return cmd_summary(Project(outdir, args.experiment))
    if args.backend == "claude-code":
        if args.command not in ("pilot", "run"):
            sys.exit("--backend claude-code supports pilot and run (batches need the API).")
        client = ClaudeCodeClient(args.claude_bin)
        if args.command == "pilot":
            return cmd_pilot(client, outdir="runs_pilot_cc")
        return cmd_run(Project(outdir, args.experiment), args.stage, client, args.workers)
    client = get_client()
    if args.command == "run":
        return cmd_run(Project(outdir, args.experiment), args.stage, client, args.workers)
    if args.command == "pilot":
        return cmd_pilot(client)
    proj = Project(outdir, args.experiment)
    if args.command == "submit":
        cmd_submit(proj, args.stage, client)
    elif args.command == "status":
        cmd_status(proj, client)
    else:
        cmd_collect(proj, args.stage, client)


if __name__ == "__main__":
    sys.exit(main())

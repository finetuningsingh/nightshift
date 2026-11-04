#!/usr/bin/env python3
"""Local (no GPU) tests: fake OpenAI-compatible server + fake verifier.

Run:  python3 -m unittest discover -s starter/recipe/tests -v
  or  python3 starter/recipe/tests/test_pipeline.py
Standard library only, like the code under test.
"""
import io, json, os, sys, tempfile, threading, time, unittest
from contextlib import redirect_stderr
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate, build_sft, summarize, origin_check  # noqa: E402

FAKE_VERIFIER = '''
def preflight(): pass
def verify_constraints(response, ground_truth):
    if ground_truth == "EMPTY": return []
    if ground_truth == "MULTI": return ["ok-token" in response, "second" in response]
    return ["ok-token" in response]
def reward(response, ground_truth, mode="fraction"):
    c = verify_constraints(response, ground_truth)
    return sum(c) / len(c) if c else 0.0
def reward_batch(responses, ground_truths, mode="fraction"):
    return [reward(r, g, mode) for r, g in zip(responses, ground_truths)]
def error_counts(): return {}
'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._json({"data": [{"id": "muse-glimmer"}]}) if self.path == "/v1/models" else self.send_error(404)
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = req["messages"][0]["content"]
        assert req["model"] == "muse-glimmer" and "max_tokens" in req
        if "SLOW" in prompt: time.sleep(3)
        content, finish, msg = "ok-token short answer", "stop", {"reasoning_content": "trace A"}
        if "TRUNCATE" in prompt: content, finish = "ok-token but cut off", "length"
        if "FAILCONS" in prompt: content = "an answer without the token"
        if "ALTREASON" in prompt: msg = {"reasoning": "trace B"}
        if "LONG" in prompt: content = "ok-token " + "x" * 500
        msg.update(role="assistant", content=content)
        self._json({"choices": [{"message": msg, "finish_reason": finish}],
                    "usage": {"completion_tokens": 7}})

def write_train(dirpath, rows):
    p = os.path.join(dirpath, "train.jsonl")
    with open(p, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    return p

def row(key, prompt, gt="GT", family="fam"):
    return {"key": key, "prompt": prompt, "ground_truth": gt, "instruction_ids": ["x"], "family": family}

def read_jsonl(path):
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]

class PipelineTest(unittest.TestCase):
    server = None
    url = None
    def url_or_skip(self):
        """Start the fake server once. Some sandboxes forbid binding any socket;
        only that environment limitation is a skip - everywhere else these run."""
        if PipelineTest.url is None:
            try:
                PipelineTest.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            except (PermissionError, OSError) as e:
                self.skipTest(f"cannot bind a local test socket in this environment: {e}")
            PipelineTest.url = f"http://127.0.0.1:{PipelineTest.server.server_address[1]}"
            threading.Thread(target=PipelineTest.server.serve_forever, daemon=True).start()
        return PipelineTest.url
    @classmethod
    def tearDownClass(cls):
        if cls.server: cls.server.shutdown()
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.verifier = os.path.join(self.d, "fake_rewards.py")
        with open(self.verifier, "w") as f: f.write(FAKE_VERIFIER)
        self.gen = os.path.join(self.d, "generations.jsonl")
        self.gstats = os.path.join(self.d, "gen_stats.json")

    def gen_args(self, train, **over):
        a = ["--base-url", self.url_or_skip(), "--train-file", train, "--out", self.gen, "--stats", self.gstats,
             "--prompt-limit", "0", "--samples", "2", "--concurrency", "4", "--max-tokens", "64",
             "--timebox-seconds", "60", "--grace-seconds", "5", "--request-timeout", "10"]
        for k, v in over.items(): a += [f"--{k}", str(v)]
        return a

    def test_generation_writes_samples_and_both_reasoning_fields(self):
        train = write_train(self.d, [row("k1", "hello"), row("k2", "ALTREASON please")])
        self.assertEqual(generate.main(self.gen_args(train)), 0)
        recs = read_jsonl(self.gen)
        self.assertEqual(len(recs), 4)
        by_key = {}
        for r in recs: by_key.setdefault(r["key"], r)
        self.assertEqual(by_key["k1"]["reasoning"], "trace A")   # reasoning_content field
        self.assertEqual(by_key["k2"]["reasoning"], "trace B")   # reasoning field
        self.assertTrue(all(r["content"] and not r["error"] for r in recs))
        self.assertEqual(json.load(open(self.gstats))["completed"], 4)

    def test_timebox_keeps_finished_and_reports_partial(self):
        rows = [row("fast", "hello")] + [row(f"s{i}", "SLOW one") for i in range(3)]
        train = write_train(self.d, rows)
        t0 = time.time()
        rc = generate.main(self.gen_args(train, samples=1, **{"timebox-seconds": 1, "grace-seconds": 0}))
        self.assertEqual(rc, 0)
        self.assertLess(time.time() - t0, 2.5)  # did not wait for the 3s sleeps
        stats = json.load(open(self.gstats))
        self.assertTrue(stats["timed_out"])
        self.assertGreaterEqual(stats["completed"], 1)
        self.assertEqual(len(read_jsonl(self.gen)), stats["completed"])

    def test_server_unreachable_fails_loudly(self):
        train = write_train(self.d, [row("k1", "hello")])
        err = io.StringIO()
        with redirect_stderr(err):
            rc = generate.main(["--base-url", "http://127.0.0.1:9", "--train-file", train,
                                "--out", self.gen, "--stats", self.gstats])
        self.assertNotEqual(rc, 0)
        self.assertIn("unreachable", err.getvalue())

    def test_prompt_file_validation(self):
        train = write_train(self.d, [{"key": "bad", "prompt": "x"}])  # no ground_truth
        with self.assertRaises(SystemExit) as cm: generate.load_prompts(train, 0, 42)
        self.assertIn("ground_truth", str(cm.exception))

    # ---------- filtering / SFT ----------
    def generations_file(self, recs):
        with open(self.gen, "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        return self.gen
    @staticmethod
    def grec(key, idx, content, finish="stop", gt="GT", reasoning="trace", prompt="do it"):
        return {"key": key, "sample_idx": idx, "prompt": prompt, "ground_truth": gt,
                "instruction_ids": ["x"], "family": "fam", "content": content,
                "reasoning": reasoning, "finish_reason": finish, "error": "", "usage": {}}
    def build(self, **over):
        out, stats = os.path.join(self.d, "sft.jsonl"), os.path.join(self.d, "sft_stats.json")
        a = ["--generations", self.gen, "--out", out, "--stats", stats,
             "--rewards-py", self.verifier, "--min-examples", "1"]
        for k, v in over.items(): a += [f"--{k}", str(v)] if v is not True else [f"--{k}"]
        return build_sft.main(a), out, stats

    def test_filtering_drops_truncated_failed_duplicate_and_caps(self):
        self.generations_file([
            self.grec("k1", 0, "ok-token a longer passing answer body"),
            self.grec("k1", 1, "ok-token a longer passing answer body"),  # exact duplicate
            self.grec("k1", 2, "ok-token mid"),
            self.grec("k1", 3, "ok-token"),                                # shortest, kept first
            self.grec("k2", 0, "ok-token cut", finish="length"),           # truncated: never kept
            self.grec("k3", 0, "no token here"),                           # constraint failed
            self.grec("k4", 0, "ok-token but no checks", gt="EMPTY"),      # vacuous: not a pass
            self.grec("k5", 0, "ok-token only one of two", gt="MULTI"),    # every constraint must pass
            self.grec("k6", 0, ""),                                        # empty
        ])
        rc, out, stats_p = self.build(**{"max-keep-per-prompt": 2})
        self.assertEqual(rc, 0)
        sft = read_jsonl(out)
        self.assertEqual(sorted(m["messages"][1]["content"] for m in sft),
                         ["ok-token", "ok-token mid"])  # cap=2, shortest first
        st = json.load(open(stats_p))
        self.assertEqual(st["dropped"], {"exact_duplicate": 1, "truncated_length": 1,
                                         "constraint_failed": 2, "no_checks_returned": 1, "empty_content": 1})
        self.assertEqual(st["sft_examples"], 2)
        # harness SFT format: messages of {role, content}, an assistant turn, no images, no reasoning by default
        for m in sft:
            self.assertEqual(set(m), {"messages"})
            self.assertEqual([x["role"] for x in m["messages"]], ["user", "assistant"])
            for x in m["messages"]:
                self.assertEqual(set(x), {"role", "content"}); self.assertTrue(x["content"])

    def test_keep_reasoning_flag_adds_reasoning_content(self):
        self.generations_file([self.grec("k1", 0, "ok-token answer", reasoning="my trace")])
        rc, out, _ = self.build(**{"keep-reasoning": True})
        self.assertEqual(rc, 0)
        self.assertEqual(read_jsonl(out)[0]["messages"][1]["reasoning_content"], "my trace")

    def test_zero_yield_and_low_yield_fail_loudly(self):
        self.generations_file([self.grec("k1", 0, "nope"), self.grec("k2", 0, "ok-token", finish="length")])
        err = io.StringIO()
        with redirect_stderr(err): rc, _, _ = self.build()
        self.assertNotEqual(rc, 0); self.assertIn("zero yield", err.getvalue())
        self.generations_file([self.grec("k1", 0, "ok-token")])
        err = io.StringIO()
        with redirect_stderr(err): rc, _, _ = self.build(**{"min-examples": 5})
        self.assertNotEqual(rc, 0); self.assertIn("MIN_EXAMPLES", err.getvalue())

    def test_verifier_preflight_and_missing_module(self):
        self.assertEqual(build_sft.main(["--preflight-only", "--rewards-py", self.verifier]), 0)
        with self.assertRaises(SystemExit):
            build_sft.main(["--preflight-only", "--rewards-py", "/nonexistent/rewards.py"])

    def test_end_to_end_and_summarize(self):
        train = write_train(self.d, [row("k1", "hello"), row("k2", "TRUNCATE me"), row("k3", "FAILCONS now")])
        self.assertEqual(generate.main(self.gen_args(train, samples=1)), 0)
        rc, out, stats_p = self.build()
        self.assertEqual(rc, 0)
        self.assertEqual(len(read_jsonl(out)), 1)  # only k1 passes and is not truncated
        timings = os.path.join(self.d, "timings.json")
        self.assertEqual(summarize.main(["--gen-stats", self.gstats, "--sft-stats", stats_p,
                                         "--out", timings, "--gen-seconds", "3"]), 0)
        t = json.load(open(timings))
        self.assertEqual(t["yield"]["sft_examples"], 1)
        self.assertEqual(t["stage_seconds"]["generate"], 3)

    def test_config_invariants(self):
        cfg = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")).read()
        self.assertIn("load_in_4bit: true", cfg)
        self.assertEqual(cfg.count("\n  - path:"), 1)       # exactly one dataset entry
        active = "\n".join(l for l in cfg.splitlines() if not l.lstrip().startswith("#"))
        self.assertNotIn("split:", active)                  # reference's 1% split must not survive
        self.assertNotIn("wandb_project:", active)          # no W&B login attempts on the GPU box
        self.assertIn("SMOKE_MAX_STEPS_ANCHOR", cfg)        # run.sh smoke substitution point

    # ---------- training-data origin rule ----------
    def origin_inputs(self):
        train_rows = [row("k1", "approved prompt one"), row("k2", "approved prompt two")]
        samples = [
            {"key": "k1", "model": "muse-glimmer", "answer": "recorded answer ",
             "reasoning": "recorded trace", "finish_reason": "stop"},
            {"key": "k2", "model": "muse-glimmer", "answer": "truncated answer",
             "reasoning": None, "finish_reason": "length"},
            {"key": "k2", "model": "some-other-model", "answer": "wrong-model answer",
             "reasoning": None, "finish_reason": "stop"},
        ]
        return train_rows, samples

    def test_origin_check_accepts_recorded_sample_exact_text(self):
        train_rows, samples = self.origin_inputs()
        good = [{"messages": [{"role": "user", "content": "approved prompt one"},
                              {"role": "assistant", "content": "recorded answer "}]}]
        self.assertEqual(origin_check.check_sft_origin(good, train_rows, samples), [])
        # optional leading system and a matching reasoning field are allowed
        with_sys = [{"messages": [{"role": "system", "content": "behave"},
                                  {"role": "user", "content": "approved prompt one"},
                                  {"role": "assistant", "content": "recorded answer ",
                                   "reasoning_content": "recorded trace"}]}]
        self.assertEqual(origin_check.check_sft_origin(with_sys, train_rows, samples), [])

    def test_origin_check_refuses_hand_written_answer_and_changed_prompt(self):
        train_rows, samples = self.origin_inputs()
        hand_written = [{"messages": [{"role": "user", "content": "approved prompt one"},
                                      {"role": "assistant", "content": "I typed this answer myself"}]}]
        self.assertTrue(origin_check.check_sft_origin(hand_written, train_rows, samples))
        changed_prompt = [{"messages": [{"role": "user", "content": "approved prompt one!"},
                                        {"role": "assistant", "content": "recorded answer "}]}]
        self.assertTrue(origin_check.check_sft_origin(changed_prompt, train_rows, samples))
        # a trimmed copy of the recorded answer is also refused (exact text or nothing)
        trimmed = [{"messages": [{"role": "user", "content": "approved prompt one"},
                                 {"role": "assistant", "content": "recorded answer"}]}]
        self.assertTrue(origin_check.check_sft_origin(trimmed, train_rows, samples))

    def test_origin_check_refuses_other_origin_violations(self):
        train_rows, samples = self.origin_inputs()
        def bad(messages): return origin_check.check_sft_origin([{"messages": messages}], train_rows, samples)
        self.assertTrue(bad([{"role": "user", "content": "approved prompt two"},
                             {"role": "assistant", "content": "truncated answer"}]))      # finish_reason length
        self.assertTrue(bad([{"role": "user", "content": "approved prompt two"},
                             {"role": "assistant", "content": "wrong-model answer"}]))    # other model's sample
        self.assertTrue(bad([{"role": "user", "content": "approved prompt one"},
                             {"role": "assistant", "content": "recorded answer ",
                              "reasoning_content": "an invented trace"}]))                # reasoning mismatch
        self.assertTrue(bad([{"role": "user", "content": "approved prompt one"},
                             {"role": "assistant", "content": "recorded answer "},
                             {"role": "user", "content": "extra turn"}]))                  # extra turn
        self.assertTrue(bad([{"role": "tool", "content": "approved prompt one"},
                             {"role": "assistant", "content": "recorded answer "}]))      # unknown role

    def test_generation_records_glimmer_samples_file(self):
        # Stubbed transport (no socket needed): exact, unmodified server text.
        train = write_train(self.d, [row("k1", "hello"), row("k2", "ALTREASON please")])
        samples_out = os.path.join(self.d, "glimmer_samples.jsonl")
        real_http = generate.http_json
        def fake_http(url, payload=None, timeout=30):
            if payload is None: return {"data": [{"id": "muse-glimmer"}]}
            prompt = payload["messages"][0]["content"]
            if "ALTREASON" in prompt:
                msg = {"role": "assistant", "content": "ok-token alt  \n", "reasoning": "trace B"}
            else:
                msg = {"role": "assistant", "content": "ok-token main", "reasoning_content": "trace A"}
            return {"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {}}
        generate.http_json = fake_http
        try:
            rc = generate.main(["--base-url", "http://stub", "--train-file", train, "--out", self.gen,
                                "--samples-out", samples_out, "--stats", self.gstats,
                                "--prompt-limit", "0", "--samples", "1", "--concurrency", "2",
                                "--timebox-seconds", "60"])
        finally:
            generate.http_json = real_http
        self.assertEqual(rc, 0)
        recs = read_jsonl(samples_out)
        self.assertEqual(len(recs), 2)
        by_key = {r["key"]: r for r in recs}
        for r in recs: self.assertEqual(set(r), {"key", "model", "answer", "reasoning", "finish_reason"})
        self.assertTrue(all(r["model"] == "muse-glimmer" for r in recs))
        self.assertEqual(by_key["k1"]["answer"], "ok-token main")
        self.assertEqual(by_key["k1"]["reasoning"], "trace A")
        self.assertEqual(by_key["k2"]["answer"], "ok-token alt  \n")  # trailing whitespace kept, never trimmed
        self.assertEqual(by_key["k2"]["reasoning"], "trace B")

    def test_build_sft_uses_samples_and_train_as_authority(self):
        self.generations_file([
            self.grec("k1", 0, "ok-token recorded  \n", reasoning="trace A", prompt="TAMPERED PROMPT"),
            self.grec("k1", 1, "ok-token never recorded", reasoning="trace X"),
        ])
        train = write_train(self.d, [row("k1", "approved prompt")])
        samples_out = os.path.join(self.d, "glimmer_samples.jsonl")
        with open(samples_out, "w") as f:
            f.write(json.dumps({"key": "k1", "model": "muse-glimmer", "answer": "ok-token recorded  \n",
                                "reasoning": "trace A", "finish_reason": "stop"}) + "\n")
        rc, out, stats_p = self.build(**{"train-file": train, "samples": samples_out})
        self.assertEqual(rc, 0)
        sft = read_jsonl(out)
        self.assertEqual(len(sft), 1)
        self.assertEqual(sft[0]["messages"][0]["content"], "approved prompt")    # train file wins over generations copy
        self.assertEqual(sft[0]["messages"][1]["content"], "ok-token recorded  \n")  # byte-identical, not trimmed
        self.assertEqual(json.load(open(stats_p))["dropped"], {"not_a_recorded_sample": 1})

if __name__ == "__main__":
    unittest.main()

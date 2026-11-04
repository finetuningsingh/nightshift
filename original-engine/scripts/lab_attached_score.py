#!/usr/bin/env python3
"""Run lab CLI commands against ONE already-running vLLM on a rental the operator owns (paired reference experiment).

    python3 scripts/lab_attached_score.py register-adapter LAB_ROOT --id ID --dir ADAPTER_DIR
    python3 scripts/lab_attached_score.py score LAB_ROOT --base-url http://127.0.0.1:18000 --rental-id RID --rate 1.12 \
        [--adapter-id ID --adapter-name NAME] [--workers 128] -- <arguments of `python -m nightshift.lab LAB_ROOT score ...`>

``score`` builds the real lab world (ledger, guard, scorer, verifiers) and replaces only the model server by ``AttachedModelServer`` and the
generator by the Inspect AI one with more workers; the command itself is the ordinary ``score`` command. Secrets are never read here.
"""
from __future__ import annotations

import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nightshift.lab import cli  # noqa: E402
from nightshift.lab.attached_server import AttachedModelServer  # noqa: E402
from nightshift.lab.contracts import ADAPTER_FILES, AdapterRef, LabConfig, LabLayout, adapter_ref_from_dir, loads_strict, read_record, verify_adapter_dir, write_record  # noqa: E402
from nightshift.lab.inspect_generator import InspectGenerator  # noqa: E402


def register_adapter(lab_root: str, adapter_id: str, src: str) -> AdapterRef:
    layout = LabLayout(Path(lab_root))
    dest = layout.adapter_dir(adapter_id)
    if dest.exists():
        raise SystemExit(f"{dest} exists already; an adapter id is never reused")
    dest.mkdir(parents=True)
    for name in ADAPTER_FILES:  # only the behaviour-defining files the ref hashes
        shutil.copyfile(Path(src) / name, dest / name)
    ref = adapter_ref_from_dir(dest, adapter_id=adapter_id)
    write_record(layout.adapter_json_path(adapter_id), ref)
    verify_adapter_dir(ref, root=layout.root)
    return ref


def run_score(args: argparse.Namespace, rest: list) -> int:
    layout = LabLayout(Path(args.lab_root))
    cfg = LabConfig.from_dict(loads_strict(layout.lab_config_path.read_bytes().decode("utf-8")))
    world = cli.build_world(layout, cfg)
    names = {}
    if args.adapter_id:
        ref = read_record(layout.adapter_json_path(args.adapter_id), AdapterRef)
        names[ref.sha256] = args.adapter_name
    server = AttachedModelServer(args.base_url, args.base_name, names, rental_id=args.rental_id, rate_usd_per_hour=args.rate)
    world = dataclasses.replace(
        world, model_server=server, generator_factory=lambda ep: InspectGenerator(ep.base_url, log_dir=layout.private_dir / "inspect_logs", workers=args.workers, timeout=args.timeout),
    )
    return cli.main([args.lab_root, "score", *rest], world=world)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register-adapter")
    r.add_argument("lab_root")
    r.add_argument("--id", required=True)
    r.add_argument("--dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("lab_root")
    s.add_argument("--base-url", required=True)
    s.add_argument("--base-name", default="muse-glimmer")
    s.add_argument("--rental-id", required=True)
    s.add_argument("--rate", type=float, required=True, help="USD per hour of the rental")
    s.add_argument("--adapter-id")
    s.add_argument("--adapter-name")
    s.add_argument("--workers", type=int, default=128)
    s.add_argument("--timeout", type=float, default=1500.0)
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if args.cmd == "register-adapter":
        ref = register_adapter(args.lab_root, args.id, args.dir)
        print(f"registered {args.id}: sha256 {ref.sha256}")
        return 0
    if bool(args.adapter_id) != bool(args.adapter_name):
        ap.error("--adapter-id and --adapter-name go together")
    return run_score(args, rest)


if __name__ == "__main__":
    sys.exit(main())

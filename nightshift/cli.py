"""Nightshift command line (spec 3.12)."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from .budget import stop_from_leases
from .config import load_config, write_demo_config
from .errors import NightshiftError
from .ledger import verify_ledger
from .pack import create_pack
from .providers import make_provider
from .report import status_text, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nightshift", description="Nightshift controller CLI")
    sub = parser.add_subparsers(dest="command")

    init = sub.add_parser("init", help="write a config file")
    init.add_argument("path", nargs="?", default="nightshift.toml")
    init.add_argument("--demo", action="store_true", help="write the CPU demo config")

    run = sub.add_parser("run", help="run from a TOML config")
    run.add_argument("config")

    demo = sub.add_parser("demo", help="run the CPU fake end-to-end demo")
    demo.add_argument("--root", default=None, help="demo directory (default: temp dir)")

    status = sub.add_parser("status", help="show run status")
    status.add_argument("run_dir")

    report = sub.add_parser("report", help="regenerate report.md for a run")
    report.add_argument("run_dir")

    ledger = sub.add_parser("ledger", help="ledger tools")
    ledger_sub = ledger.add_subparsers(dest="ledger_command")
    verify = ledger_sub.add_parser("verify", help="verify a run ledger")
    verify.add_argument("run_dir")

    stop = sub.add_parser("stop", help="destroy every live owned rental in a lease file")
    stop.add_argument("--leases", required=True)
    stop.add_argument("--provider", required=True)
    stop.add_argument("--host", default="")
    stop.add_argument("--user", default="")
    return parser


def _ledger_path(run_dir: str) -> Path:
    path = Path(run_dir)
    return path if path.suffix == ".jsonl" else path / "ledger.jsonl"


def _print_progress(message: str) -> None:
    print(f"nightshift: {message}", flush=True)


def run_config(path: str | Path) -> dict:
    from .loops import run_pipeline

    config = load_config(path)
    label = " [fake]" if config.fake else ""
    _print_progress(f"phase=start run={config.run_name} config={path}{label}")
    summary = run_pipeline(config, progress=_print_progress)
    _print_progress(
        f"phase=done report={config.output_dir / 'report.md'}{label}"
    )
    return summary


def run_demo(root: str | Path | None = None) -> Path:
    """Init + run the CPU demo. Needs no secrets, network, or GPU."""
    demo_root = (
        Path(root) if root is not None
        else Path(tempfile.mkdtemp(prefix="nightshift-demo-"))
    )
    demo_root.mkdir(parents=True, exist_ok=True)
    create_pack(demo_root / "pack")
    config_path = write_demo_config(
        demo_root / "nightshift.toml",
        output_dir="run",
        budget_file="run/budget.json",
        pack_dir="pack",
    )
    _print_progress(f"phase=demo root={demo_root} [fake]")
    run_config(config_path)
    run_dir = demo_root / "run"
    result = verify_ledger(_ledger_path(str(run_dir)))
    if not result.ok:
        raise NightshiftError(
            f"demo ledger failed verification at seq {result.broken_seq}: {result.message}"
        )
    _print_progress(
        f"phase=demo-done run_dir={run_dir} report={run_dir / 'report.md'} "
        f"ledger=ok ({result.count} events) [fake]"
    )
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            target = write_demo_config(args.path)
            load_config(target)
            kind = "CPU demo config (fake provider/agents, bundled toy data)"
            if not args.demo:
                kind += "; pass --demo for the same demo defaults, edit before any real run"
            print(f"wrote {kind}: {target}")
            return 0
        if args.command == "run":
            run_config(args.config)
            return 0
        if args.command == "demo":
            run_demo(args.root)
            return 0
        if args.command == "status":
            print(status_text(args.run_dir))
            return 0
        if args.command == "report":
            target = write_report(args.run_dir)
            print(f"wrote report {target}")
            return 0
        if args.command == "ledger" and args.ledger_command == "verify":
            result = verify_ledger(_ledger_path(args.run_dir))
            if result.ok:
                print(f"ledger ok: {result.count} events")
                return 0
            print(f"ledger BROKEN at seq {result.broken_seq}: {result.message}", file=sys.stderr)
            return 1
        if args.command == "stop":
            provider = make_provider(
                args.provider, {"host": args.host, "user": args.user}
            )
            summary = stop_from_leases(args.leases, provider)
            label = " [fake]" if summary["fake"] else ""
            print(
                f"destroyed {len(summary['destroyed'])} owned rental(s){label}: "
                f"{', '.join(summary['destroyed']) or 'none'}"
            )
            if summary.get("stop_failed"):
                print(
                    f"error: destruction NOT confirmed for "
                    f"{', '.join(summary['stop_failed'])}; "
                    f"lease(s) kept live for retry{label}",
                    file=sys.stderr,
                )
                return 1
            return 0
        parser.print_help()
        return 2
    except NightshiftError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

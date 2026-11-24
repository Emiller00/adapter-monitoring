"""Verify saved GLNet evaluation evidence."""
import argparse
import json
import sys
from pathlib import Path

from .audit import audit
from .generate import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    audit_command = commands.add_parser("audit", help="Recompute metrics from a saved evidence bundle")
    audit_command.add_argument("bundle", nargs="?", type=Path)
    audit_command.add_argument("--json", action="store_true", help="Print machine-readable results")
    data_command = commands.add_parser("generate-data", help="Regenerate the synthetic prompt datasets")
    data_command.add_argument("output", type=Path)
    args = parser.parse_args()
    bundled = Path(__file__).parent / "evidence"
    try:
        if args.command == "generate-data":
            report = generate(bundled / "recipe.json", args.output)
            print(json.dumps(report, indent=2))
            return 0
        report = audit(getattr(args, "bundle", None) or bundled)
        if getattr(args, "json", False):
            print(json.dumps(report, indent=2))
        else:
            for run in report:
                print(f"{run['run']} ({run['evaluation_blocks']} evaluation blocks)")
                for name, counts in run["monitors"].items():
                    detected, flagged = counts["detected"], counts["false_alarms"]
                    print(f"  {name}: detected {detected[0]}/{detected[1]}, "
                          f"false alarms {flagged[0]}/{flagged[1]}")
                if "qualification" in run:
                    q = run["qualification"]
                    print(f"  Qualification failures: {q['all_failures']}/{q['all_actors']} overall, "
                          f"{q['evaluation_failures']}/{q['evaluation_actors']} held out")
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

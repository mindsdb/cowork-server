"""Compare two completed runs of the same bounded workload and capacity."""
import argparse
import json
from pathlib import Path

from scripts.performance.profile_answers import compare_reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scope", choices=["server", "full-stack"], default="server")
    args = parser.parse_args()
    print(json.dumps(compare_reports(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()), scope=args.scope), indent=2))


if __name__ == "__main__":
    main()

"""Mining classifier CLI.

    uv run python -m bantay.ml features            # cache features for every region (training year)
    uv run python -m bantay.ml train               # cross-validate, train, write ML_REPORT.md
    uv run python -m bantay.ml predict --year 2026 # candidates for the Northern Luzon test regions
"""

import argparse

from . import HANSEN, ML_DATA, REPORT, regions
from .features import load_or_build


def main() -> None:
    parser = argparse.ArgumentParser(prog="bantay.ml")
    sub = parser.add_subparsers(dest="cmd", required=True)
    feat = sub.add_parser("features", help="build and cache feature stacks")
    feat.add_argument("names", nargs="*", help="region names (default: all)")
    feat.add_argument("--year", type=int, default=None)
    sub.add_parser("train", help="cross-validate, train and write the scorecard")
    pred = sub.add_parser("predict", help="write unreviewed candidate patches")
    pred.add_argument("names", nargs="*", help="region names (default: all test regions)")
    pred.add_argument("--year", type=int, default=2026)
    args = parser.parse_args()

    regs = regions()
    everything = {**regs["train"], **regs["test"]}
    if args.cmd == "features":
        from .train import YEAR
        for name in args.names or list(everything):
            if name not in everything:
                parser.error(f"unknown region {name!r}")
            load_or_build(everything[name], args.year or YEAR, ML_DATA, HANSEN)
            print(f"{name}: features cached")
    elif args.cmd == "train":
        from .train import run
        run(regs, REPORT)
    else:
        from .predict import run
        for name in args.names or list(regs["test"]):
            if name not in everything:
                parser.error(f"unknown region {name!r}")
            run(everything[name], args.year)


if __name__ == "__main__":
    main()

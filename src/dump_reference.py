"""Dump reference outputs of the CURRENT parser build to a JSON file.

Run this against the last-known-good build (e.g. before a rearchitecture),
then verify the new build with compare_reference.py.

Usage: python src/dump_reference.py [parquet_path] [output_json]
"""

import json
import sys
import time

from reference_lib import build_reference

DEFAULT_PARQUET = "2013-07-train-00000-of-00001.parquet"
DEFAULT_OUTPUT = "reference_2013-07.json"


def main():
    parquet = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PARQUET
    output = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    t0 = time.time()
    ref = build_reference(parquet)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(ref, f, indent=1, ensure_ascii=False)

    main_part = ref["main"]
    print(f"Wrote {output} in {time.time() - t0:.1f}s")
    print(
        f"  games={main_part['num_games']} invalid={main_part['num_invalid']} "
        f"valid_moves={main_part['valid_moves_total']} "
        f"valid_positions={main_part['valid_positions_total']}"
    )


if __name__ == "__main__":
    main()

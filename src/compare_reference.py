"""Compare the CURRENT parser build's outputs against a saved reference.

Usage: python src/compare_reference.py [reference_json]

Exits nonzero and prints mismatches if the outputs differ.
"""

import json
import sys
import time

from reference_lib import build_reference, diff_dicts

DEFAULT_REFERENCE = "reference_2013-07.json"


def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REFERENCE

    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)

    t0 = time.time()
    new = build_reference(ref["parquet"], subset_n=ref["subset_n"])
    print(f"Recomputed summary in {time.time() - t0:.1f}s")

    mismatches = diff_dicts(ref, new)
    if mismatches:
        print(f"MISMATCH: {len(mismatches)} difference(s):")
        for m in mismatches[:50]:
            print(f"  {m}")
        if len(mismatches) > 50:
            print(f"  ... and {len(mismatches) - 50} more")
        sys.exit(1)

    print("OK: new output matches reference")


if __name__ == "__main__":
    main()

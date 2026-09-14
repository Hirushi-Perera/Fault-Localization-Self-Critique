"""
Look up the real (gold) developer fix for a SWE-bench Verified bug, so you can compare
it by eye against whatever Agentless/Qwen produced.

Usage:
    python check_gold_patch.py <instance_id>

Example:
    python check_gold_patch.py django__django-15629
"""
import sys

from datasets import load_dataset

if len(sys.argv) != 2:
    print("Usage: python check_gold_patch.py <instance_id>")
    sys.exit(1)

instance_id = sys.argv[1]

ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
row = next((r for r in ds if r["instance_id"] == instance_id), None)

if row is None:
    print(f"instance_id '{instance_id}' not found in SWE-bench_Verified")
    sys.exit(1)

print(f"=== Gold patch for {instance_id} ===\n")
print(row["patch"])

import sys
import pandas as pd

input_id = sys.argv[1] if len(sys.argv) > 1 else "p01"
d = pd.read_json("results/run1/runs.jsonl", lines=True)
d = d[(d.input_id == input_id) & (d.rep == 0)].sort_values("variant_id")
for r in d.itertuples():
    print(f"[{r.variant_id}] {r.words} words, {r.output_tokens} output tokens")
    print(r.output.strip(), "\n")
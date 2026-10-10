# python -m src.reasoning.soup RUNS FOLD
import sys
from pathlib import Path

import peft

from .writer import load

runs, fold_name = sys.argv[1], sys.argv[2]
run_dir = Path(f"runs/soup/{runs}/fold_{fold_name}")
paths = [f"runs/{runs}/seed{s}/fold_{fold_name}/final_adapter" for s in range(3)]

base, _ = load(nf4=False)
model = peft.PeftModel.from_pretrained(base, paths[0], adapter_name="s0")
model.load_adapter(paths[1], adapter_name="s1")
model.load_adapter(paths[2], adapter_name="s2")
model.add_weighted_adapter(["s0", "s1", "s2"], [1 / 3] * 3, adapter_name="merged", combination_type="cat")
model.save_pretrained(str(run_dir), selected_adapters=["merged"])
(run_dir / "merged").rename(run_dir / "final_adapter")
print("soup", run_dir / "final_adapter")

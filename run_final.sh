#!/bin/sh
# ./run_final.sh [public|private] [REPEATS]
set -e
TEST=${1:-private}
RUN=sft_x${2:-3}nf4_r32

bash install_unsloth.sh
bash install_vllm.sh
for split in train $TEST; do
  [ -s runs/charts/${split}_8bz.jsonl ] || .venv-vllm/bin/python -m src.charts $split
done
for s in 0 1 2; do python -m src.main all $s ${2:-3}; done
python -m src.reasoning.soup $RUN all
python -m src.reasoning.predict $TEST runs/soup/$RUN/fold_all

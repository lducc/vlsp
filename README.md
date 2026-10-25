# VLSP 2026 Numerical Reasoning QA

Team Arcane's system for the VLSP 2026 shared task on multimodal numerical reasoning over Vietnamese
financial reports. Given a report with text, HTML tables and chart images, it writes the reasoning
program that answers a question about it.

Paper: [Retrieval-Augmented Fine-Tuning for Vietnamese Multimodal Numerical Reasoning](paper/main.pdf).
Chart cards: [lducc/vlsp2026-chart-cards](https://huggingface.co/datasets/lducc/vlsp2026-chart-cards).

## Results

Second by Program Accuracy (PA) on both the public and the private leaderboard, with a 4B writer and one
writer pass per question.

| Team | Public rank | Public PA | Private rank | Private PA | Private EA |
|---|---:|---:|---:|---:|---:|
| thanhducsvbg | 3 | 74.73 | 1 | 79.17 | 89.33 |
| **Arcane (ours)** | **2** | **75.16** | **2** | **78.66** | **87.61** |
| Taekz | 9 | 4.14 | 3 | 76.42 | 87.44 |
| bachanglinhngulam | 1 | 75.60 | 4 | 74.35 | 86.92 |
| rudeus | 5 | 67.10 | 5 | 73.32 | 85.71 |
| HUNTER | 4 | 69.72 | 6 | 72.46 | 83.30 |

Our variants on the test sets:

| Combination | Epochs | Writer passes | Public PA | Private PA | Private EA |
|---|---:|---:|---:|---:|---:|
| Adapter soup (submitted) | 3 | 1 | 75.16 | 78.66 | 87.61 |
| Adapter soup | 2 | 1 | 73.20 | 77.45 | 88.47 |
| Sequence ensemble | 2 | 3 | 72.33 | 77.28 | 88.30 |

On held-out training reports, fine-tuning raises PA from 45.6 (base Qwen3.5-4B with the same ten BM25
examples) to 73.0, and BM25 examples beat random ones by 3.7 points.

## Example

A made-up report contains this chart as `Image 1`:

![Example chart](docs/example_chart.png)

Question: *Doanh thu năm 2024 tăng bao nhiêu phần trăm so với năm 2023?*
(By what percentage did revenue grow from 2023 to 2024?)

Qwen3-VL-8B turns the chart into a label card:

```json
{"title": "Kết quả kinh doanh 2021-2024", "series_labels": ["Doanh thu", "Lợi nhuận sau thuế"], "x_axis_labels": ["2021", "2022", "2023", "2024"]}
```

The writer reads the card, the report text and tables, and 10 similar question/program pairs from other
reports, then writes:

```text
chart_at(Image 1; Doanh thu; 2023; none); chart_at(Image 1; Doanh thu; 2024; none);
subtract(#1; #0); divide(#2; #0); multiply(#3; 100)
```

`#k` is the result of step k. The program selects chart cells by label, so the card carries labels only.

## Method

BM25 over question text picks the 10 examples. Qwen3.5-4B is fine-tuned with NF4 QLoRA on the program
tokens, and three runs that differ only in data order are averaged into one adapter:

![Adapter soup and sequence ensemble](docs/soup.png)

## Quickstart

Request the data from VLSP through the
[task page](https://vlsp.org.vn/vlsp2026/eval/vitnumchart) and place it as follows:

```text
VLSP/train.json  VLSP/train_images/
public_test.json  public_test_images/
private_test.json  private_test_images/
```

```sh
./run_final.sh private    # writes runs/soup/sft_x3nf4_r32/fold_all/private_hf/results.zip
```

Use `./run_final.sh public` for the public split. `./run_final.sh private 2` trains on two copies of the
training set instead of three.

## Stages

```sh
.venv-vllm/bin/python -m src.charts train|public|private        # chart cards
python -m src.main FOLD SEED [REPEATS]                           # train one seed, and score it if FOLD is 0-4
python -m src.reasoning.soup sft_x3nf4_r32 FOLD                  # average seeds 0-2
python -m src.reasoning.predict FOLD|public|private RUN_DIR      # greedy decoding and results.zip
python -m src.reasoning.seq_ensemble public|private sft_x2nf4_r32
```

`seq_ensemble` is the alternative to the soup. Each seed scores the three seeds' greedy programs, and the
program with the highest summed log-probability wins.

Decode NF4-trained adapters on the NF4 base. If `.env` exists (see
`.env.example`), each finished run is uploaded to a private Hugging Face repo.

## Code

```text
src/data.py                     loading, program parser, document-grouped folds
src/charts.py                   chart cards (vLLM)
src/main.py                     training entry point
src/reasoning/prompts.py        BM25 examples, prompt, chat template, response parsing
src/reasoning/program.py        Program Accuracy scorer
src/reasoning/writer.py         training, loading, decoding
src/reasoning/soup.py           seed averaging
src/reasoning/predict.py        prediction and submission zip
src/reasoning/seq_ensemble.py   sequence ensemble
```

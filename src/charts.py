# .venv-vllm/bin/python -m src.charts train|public|private
import json
import sys
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams


def parse_card(response):
    start, end = response.find("{"), response.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        card = json.loads(response[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(card, dict) or not isinstance(card.get("title", ""), str):
        return None
    for key in ("series_labels", "x_axis_labels"):
        values = card.get(key)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            return None
    return card


def main():
    split = sys.argv[1]
    images = Path("VLSP/train_images" if split == "train" else f"{split}_test_images")
    output = Path(f"runs/charts/{split}_8bz.jsonl")
    done = set()
    if output.exists():
        done = {json.loads(line)["image"] for line in output.open(encoding="utf-8")}
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(images.glob("*.png"))
    prompt = """Read this Vietnamese financial chart carefully.
Return only one JSON object with exactly these keys:
{
  "title": "visible chart title or empty string",
  "series_labels": ["every visible legend or series label"],
  "x_axis_labels": ["every visible x-axis/category/time label in display order"]
}
Copy visible text exactly, including Vietnamese diacritics, punctuation, spaces, capitalization, and date formatting.
Do not translate, normalize, correct, infer, explain, or include numeric y-axis tick values.
Use [] when a label type is absent."""

    model_id, revision = "Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    llm = LLM(model=model_id, revision=revision, dtype="bfloat16", max_model_len=7168, max_num_seqs=2,
              gpu_memory_utilization=0.95, limit_mm_per_prompt={"image": 1, "video": 0})
    params = SamplingParams(temperature=0, max_tokens=1024)

    todo = [p for p in paths if p.name not in done]
    conversations = []
    for path in todo:
        image = Image.open(path).convert("RGB")
        image = image.resize((image.width * 2, image.height * 2), Image.Resampling.LANCZOS)
        conversations.append([{"role": "user", "content": [{"type": "image_pil", "image_pil": image},
                                                            {"type": "text", "text": prompt}]}])
    outs = llm.chat(conversations, params)

    with output.open("a", encoding="utf-8") as f:
        for path, out in zip(todo, outs, strict=True):
            response = out.outputs[0].text
            card = parse_card(response)
            f.write(json.dumps({"image": path.name, "raw": response, "parsed": card}, ensure_ascii=False) + "\n")
            print(f"{path.name}: {'ok' if card else 'malformed'}", flush=True)


if __name__ == "__main__":
    main()

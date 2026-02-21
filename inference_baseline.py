# inference_baseline.py
import os
import time
import json
import shutil
import traceback

import torch
from PIL import Image
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor
import re
# -------------------------------
#       "file": "0000.png",
#       "score": 3.0,
#       "time_sec": 37.57,
#       "reasoning": "
# -------------------------------
# Configuration (modify as needed)
# -------------------------------
model_dir = "google/medgemma-4b-it"
image_dir = "./data/LDCTiqa_png"   # Path for storing 1000 PNG images
output_dir = "./inferenceOut"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "results.json")

batch_size = 2  # frequency of saving
start_index = 0 


user_prompt_text = (
    "Please analyze the image quality of this CT image and give a score from 0 to 4 (decimals are acceptable, e.g., 1.5, 2.8, 3.0). "
    "Evaluate only the image quality, without discussing the CT image's condition. Explain your basis and reasoning for this assessment:\n"
)


print("Loading models and processors（float32）...")
processor = Gemma3Processor.from_pretrained(model_dir)
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float32
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
model.to(device)
model.eval()


def extract_score_from_text(text):

    if not text:
        return None

    # **3.5**
    m = re.search(r'\*\*\s*([0-4](?:\.\d+)?)\s*\*\*', text)
    if m:
        return round(float(m.group(1)), 1)

    # 3.5/4
    m = re.search(r'([0-4](?:\.\d+)?)\s*/\s*4', text)
    if m:
        return round(float(m.group(1)), 1)

    # 3.5 out of 4
    m = re.search(r'([0-4](?:\.\d+)?)\s+(?:out of|out_of)\s+4', text, flags=re.I)
    if m:
        return round(float(m.group(1)), 1)

    # "score of 3.5"
    m = re.search(r'(?:score[:\s]*|score\s+of\s+|I would give|I\'d give)\s*([0-4](?:\.\d+)?)', text, flags=re.I)
    if m:
        return round(float(m.group(1)), 1)

    # fallback
    for mm in re.finditer(r'([0-4](?:\.\d+)?)', text):
        try:
            val = float(mm.group(1))
            if 0.0 <= val <= 4.0:
                return round(val, 1)
        except:
            continue

    return None


def clean_reasoning(raw_text):
    if not raw_text:
        return None

    idx = raw_text.lower().find("model")
    if idx != -1:
        reasoning = raw_text[idx+5:].strip()
    else:
        reasoning = raw_text.strip()

    reasoning = reasoning.strip()

    return reasoning

def infer_image(image_path):

    res = {
        "file": os.path.basename(image_path),
        "score": None,
        "time_sec": None,
        "reasoning": None,
        "error": None
    }
    try:

        img = Image.open(image_path)

        try:
            if getattr(img, "n_frames", 1) > 1:
                img.seek(0)
        except Exception:
            pass
        image = img.convert("RGB")
        

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt_text},
                {"type": "image"}
            ]}
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        

        inputs = processor(text=prompt, images=[image], return_tensors="pt")

        for k, v in inputs.items():
            inputs[k] = v.to(device)


        start = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                top_p=0.9,
                do_sample=True #To disable sampling, set it to false.
            )
        elapsed = time.time() - start


        raw_text = processor.decode(output_ids[0], skip_special_tokens=True)
        reasoning = clean_reasoning(raw_text)
        

        score_val = extract_score_from_text(reasoning)


        res["score"] = score_val if score_val is None else float(round(score_val, 1))
        res["time_sec"] = round(elapsed, 2)
        res["reasoning"] = reasoning


        del inputs, output_ids
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        res["error"] = f"{repr(e)}\n{traceback.format_exc()}"
        res["time_sec"] = None

    return res

def main():
    all_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
    total = len(all_files)
    if total == 0:
        print("No PNG image found. Exit.")
        return
    print(f"{total} images were found. batch_size={batch_size}")

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_batches = json.load(f)
            completed = sum(len(b) for b in all_batches)
            print(f"An existing result file has been detected. all_batches {len(all_batches)}，Completed {completed}")
        except Exception as e:
            print("Failed to read an existing result file; will start from the beginning. Error:", e)
            all_batches, completed = [], 0
    else:
        all_batches, completed = [], 0


    start_idx = max(start_index, completed)
    print(f"Start processing from index {start_idx}")

    total_batches = (total + batch_size - 1) // batch_size
    for batch_idx in range(start_idx // batch_size, total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, total)
        batch_files = all_files[batch_start:batch_end]


        already_done = max(0, completed - batch_start)
        if already_done >= len(batch_files):
            print(f"Skip batch {batch_idx+1} (already completed)")
            continue

        to_process = batch_files[already_done:]

        print(f"\nProcessing number {batch_idx+1}/{total_batches} ,  {batch_start} ~ {batch_end-1}，Pending processing {len(to_process)}")

        batch_results = []

        if already_done > 0:

            if batch_idx < len(all_batches):

                saved = all_batches[batch_idx]
                batch_results.extend(saved)
            else:

                pass
            
            

        for fname in to_process:
            fpath = os.path.join(image_dir, fname)
            print(f"  processing {fname} ...", end="", flush=True)

            res = infer_image(fpath)
            res["file"] = fname
            batch_results.append(res)
            completed += 1
            print(f" completed | score={res['score']} time={res['time_sec']}s error={'YES' if res['error'] else 'NO'}")


        if batch_idx < len(all_batches):

            all_batches[batch_idx] = batch_results
        else:
            all_batches.append(batch_results)


        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(all_batches, f, ensure_ascii=False, indent=2)
        shutil.move(tmp_path, output_path)
        print(f"✅ saved {batch_idx+1} to {output_path} （completed {completed}/{total}）")

    print("\nAll processing completed.")

if __name__ == "__main__":
    main()

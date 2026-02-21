#这是一个可以运行的经过微调后的权重的脚本
import os
import time
import json
import shutil
import traceback
import re
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, PeftConfig

#采用微调后的权重执行测试集任务。任务来源是一个JSON文件 (test_json_path)，只对这个JSON文件中的明确列出的（100张）图片进行推理。



# ======================================================
#                路径与基本设置（仅需修改这里）
# ======================================================
base_model_id = "google/medgemma-4b-it"

lora_dir = "./medgemma_sft_output/checkpoint-150"

image_dir = "./data/LDCTiqa_png"

output_dir = "./inferenceOut"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "detailed_FTinference_results.json") 
test_json_path = "./data/testData.json" # This is the path to the JSON file where the testset is stored.

batch_size = 1
start_index = 0

prompt_text = (
"""
Evaluate the technical and diagnostic quality of this CT image. Predict a score from 0 (very poor) to 4 (excellent), and explain your reasoning clearly. Provide both the numerical score and the explanation in your answer.

"""
)



processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)


model = AutoModelForImageTextToText.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map=None
)


if getattr(model.config, "vocab_size", None) is None:
    model.config.vocab_size = processor.tokenizer.vocab_size
    print(f"✅ Fixed vocab_size = {model.config.vocab_size}")


print("Attempt to load the LoRA model...")
try:
    peft_cfg = PeftConfig.from_pretrained(lora_dir)
    model = PeftModel.from_pretrained(model, lora_dir)
    print("✅ Successfully loaded LoRA weights")
except Exception as e:
    print(f"⚠️ LoRA loading failed: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
model = model.to(device, dtype=torch.float32)
model.eval()


def extract_score_from_text(text):
    if not text:
        return None

    #  **3.5**
    m = re.search(r'\*\*\s*([0-4](?:\.\d+)?)\s*\*\*', text)
    if m: return round(float(m.group(1)), 1)
    # 3.5/4
    m = re.search(r'([0-4](?:\.\d+)?)\s*/\s*4', text)
    if m: return round(float(m.group(1)), 1)
    # score of 3.5
    m = re.search(r'(?:score[:\s]*|score\s+of\s+|I would give|I\'d give)\s*([0-4](?:\.\d+)?)', text, flags=re.I)
    if m: return round(float(m.group(1)), 1)
    # fallback
    for mm in re.finditer(r'([0-4](?:\.\d+)?)', text):
        try:
            val = float(mm.group(1))
            if 0.0 <= val <= 4.0:
                return round(val, 1)
        except:
            continue
    return None


def infer_image(image_path):

    res = {"file": os.path.basename(image_path), "score": None, "time_sec": None, "reasoning": None, "error": None}
    try:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image"}
            ]}
        ]
        multimodal_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=multimodal_prompt, images=[image], return_tensors="pt")

        for k in inputs:
            inputs[k] = inputs[k].to(device, dtype=torch.float32 if k == "pixel_values" else torch.long)

        start = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, #sample or not
                temperature=0, 
                top_p=1.0  
            )
        elapsed = time.time() - start

        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, input_len:]
        raw_text = processor.decode(generated, skip_special_tokens=False)
        clean_text = raw_text.split("<end_of_turn>")[0].strip()

        res["score"] = extract_score_from_text(clean_text)
        res["time_sec"] = round(elapsed, 2)
        res["reasoning"] = clean_text

        del inputs, output_ids
        if device == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        res["error"] = f"{repr(e)}\n{traceback.format_exc()}"
    return res


def main():

    if not os.path.exists(test_json_path):
        print(f"Error: Test image list file not found: {test_json_path}")
        return

    with open(test_json_path, "r", encoding="utf-8") as f:
        test_images_data = json.load(f)


    all_files_to_infer = sorted(list(test_images_data.keys()))
    total = len(all_files_to_infer)
    if total == 0:
        print("No images were found in the test image list. Exit.")
        return
    print(f" {test_json_path}. A total of {total} images were found. batch_size={batch_size}")


    final_results_list = []
    completed_files_set = set() 

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                final_results_list = json.load(f)

            for res_item in final_results_list:
                if "file" in res_item:
                    completed_files_set.add(res_item["file"])
            
            completed_count = len(completed_files_set)
            print(f"Results have been detected; a total of {completed_count} images have been completed.")
        except Exception as e:
            print(f"⚠️ Loading an existing result file failed ({e}), and will start from scratch.")
            final_results_list = []
            completed_files_set = set()
            completed_count = 0
    else:
        final_results_list = []
        completed_files_set = set()
        completed_count = 0


    remaining_files = [f for f in all_files_to_infer if f not in completed_files_set]
    total_remaining = len(remaining_files)
    if total_remaining == 0:
        print("All specified images have been completed.")
        return
    
    print(f"The remaining {total_remaining} images need to be processed.")

    total_batches = (total_remaining + batch_size - 1) // batch_size

    current_processed_count = 0 

    for batch_idx in range(total_batches):
        batch_start_idx = batch_idx * batch_size
        batch_end_idx = min((batch_idx + 1) * batch_size, total_remaining)
        

        batch_files = remaining_files[batch_start_idx:batch_end_idx]
        
        print(f"\n🚀 processing {batch_idx+1}/{total_batches}  ({batch_start_idx}-{batch_end_idx-1})")

        for fname in batch_files:
            path = os.path.join(image_dir, fname)
            if not os.path.exists(path):
                print(f"  ➤ Warning: Image file {fname} was not found in {image_dir}. Skip.")
                error_res = {"file": fname, "score": None, "time_sec": None, "reasoning": None, "error": "Image file not found"}
                final_results_list.append(error_res)
                current_processed_count += 1
                continue

            print(f"  ➤ processing {fname} ...", end="", flush=True)
            res = infer_image(path)
            final_results_list.append(res)
            
            status_emoji = '❌' if res['error'] else '✅'
            score_display = res['score'] if res['score'] is not None else "N/A"
            print(f" completed | score={score_display} | {res['time_sec']}s | {status_emoji}")
            
            current_processed_count += 1

        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(final_results_list, f, ensure_ascii=False, indent=2)
        shutil.move(tmp_path, output_path)
        print(f"✅ The results for the current batch have been saved (a total of {len(final_results_list)} images have been completed).")

    print("\n🎯 All reasoning is complete. The results are saved in:", output_path)

if __name__ == "__main__":
    main()
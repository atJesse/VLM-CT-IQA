import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from datasets import load_dataset, Image as ImageFeature, Value, Features
from PIL import Image
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from typing import Any, Dict, List
import os
import json 
from transformers import TrainerCallback, TrainerState, TrainerControl

# ============================================================
# ✅ 成功的训练脚本 在脚本开头定义了要保存指标的 TXT 文件路径，确保它位于 output_dir 下。目录下会生成一个 training_metrics.txt 文件。定义epoch为3试试
# ============================================================

# --- 1. This is the configuration and path; you can check and modify it here. ---
dataset_path = "./data" #This is the path where train.jsonl and val.jsonl are located.
train_file = os.path.join(dataset_path, "train.jsonl")
val_file = os.path.join(dataset_path, "val.jsonl")
model_id = "google/medgemma-4b-it"
output_dir = "./medgemma_sft_output"

metrics_log_file = os.path.join(output_dir, "training_metrics.txt")


os.makedirs(output_dir, exist_ok=True)


print(f"--- Configuration ---")
print(f"Dataset path: {dataset_path}")
print(f"Training file: {train_file}")
print(f"Validation file: {val_file}")
print(f"Model ID: {model_id}")
print(f"Output directory: {output_dir}")
print(f"Metrics log file: {metrics_log_file}")
print(f"---------------------")

# --- 2. Check GPU support ---
if not torch.cuda.is_available():
    raise SystemError("No CUDA GPU found. This script requires a GPU for training.")

gpu_capability = torch.cuda.get_device_capability()
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"GPU Capability (Major.Minor): {gpu_capability[0]}.{gpu_capability[1]}")
print(f"GPU VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

if gpu_capability[0] < 8:
    print("WARNING: GPU does not natively support bfloat16 (compute capability < 8). "
          "Training might be slower or encounter issues. "
          "Consider using a GPU with compute capability 8.0 or higher for optimal bfloat16 performance.")

# --- 3. Loading Datasets ---
print(f"\n--- Loading Datasets ---")
try:
    dataset_features = Features({
        "image": ImageFeature(),
        "instruction": Value("string"),
        "output": Value("string")
    })

    data = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file},
        features=dataset_features,
    )
    print(f"Successfully loaded dataset structure.")
    print(f"Train dataset size: {len(data['train'])}")
    print(f"Validation dataset size: {len(data['validation'])}")

    print("\n--- Example Training Data Sample ---")
    first_train_example = data['train'][0]
    if isinstance(first_train_example['image'], Image.Image):
        print(f"Image loaded as PIL Image: YES")
        print(f"Image format: {first_train_example['image'].format}, size: {first_train_example['image'].size}")
    else:
        print(f"Image not loaded as PIL Image. Type: {type(first_train_example['image'])}")
        print(f"Image raw content: {first_train_example['image']}")

    print(f"Instruction: {first_train_example['instruction'][:100]}...")
    print(f"Output: {first_train_example['output'][:100]}...")

except Exception as e:
    print(f"ERROR: Failed to load dataset. Please check file paths and format.")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# --- 4. Loading Model and Processor ---
print(f"\n--- Loading Model and Processor ---")
model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
)

model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
    bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    llm_int8_enable_fp32_cpu_offload=True,
)

try:
    print(f"Loading model '{model_id}' with quantization config...")
    custom_device_map = {
        "": 0,
        "lm_head": "cpu",
        "model.embed_tokens": "cpu",
    }
    print(f"Using custom device_map: {custom_device_map}")

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs,
        device_map=custom_device_map,
    )
    print(f"Model loaded successfully. Model type: {type(model)}")
    print(f"Model device: {model.device}")

    print("\n--- Model Layer Device Mapping ---")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            print(f"  {name}: {module.weight.device}")
        elif hasattr(module, 'device'):
            print(f"  {name}: {module.device}")
        elif isinstance(module, torch.nn.Module) and len(list(module.parameters())) > 0:
            for p_name, p in module.named_parameters():
                print(f"  {name}.{p_name}: {p.device}")
                break
    print("----------------------------------")

    print(f"Loading processor for '{model_id}'...")
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Processor loaded successfully. Tokenizer padding side before: {processor.tokenizer.padding_side}")

    processor.tokenizer.padding_side = "right"
    print(f"Tokenizer padding side set to: {processor.tokenizer.padding_side}")

    print(f"Special tokens map: {processor.tokenizer.special_tokens_map}")
    print(f"Image token ID (boi_token): {processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.special_tokens_map['boi_token'])}")
    print(f"Pad token ID: {processor.tokenizer.pad_token_id}")

except Exception as e:
    print(f"ERROR: Failed to load model or processor. Please check model_id and internet connection.")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# --- 5. Setting up LoRA Configuration ---
print(f"\n--- Setting up LoRA Configuration ---")
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
print(f"LoRA config: {peft_config}")

# --- 6. Custom Data Collator ---
def collate_fn(examples: List[Dict[str, Any]]):

    images_for_processor = [] 
    texts_for_processor = [] 

    # print(f"--- Data Collator Batch Start ({len(examples)} examples) ---") 
    for i, example in enumerate(examples):

        if not isinstance(example["image"], Image.Image):
            # print(f"WARNING: Example {i} image is not PIL.Image. Type: {type(example['image'])}. Attempting conversion.") 
            try:
                example["image"] = Image.open(example["image"]).convert("RGB")
                # print(f"Example {i}: Manually loaded image successfully.") 
            except Exception as e:
                print(f"ERROR: Could not load image for example {i} from path. Error: {e}")
                continue


        images_for_processor.append([example["image"].convert("RGB")]) 
        
        current_messages = [
            {"role": "user", "content": [
                {"type": "text", "text": example["instruction"]},
                {"type": "image"}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": example["output"]},
            ]},
        ]
        
        formatted_text = processor.apply_chat_template(
            current_messages, 
            add_generation_prompt=True, 
            tokenize=False 
        )
        texts_for_processor.append(formatted_text)

        # print(f"Example {i}: Image size {example['image'].size}, Formatted text length: {len(formatted_text)}") 

    if not texts_for_processor:
        # print("WARNING: No valid examples processed in this batch. Returning empty batch.") 
        return {}

    # Tokenize the texts and process the images
    # print(f"Calling processor with {len(texts_for_processor)} texts and {len(images_for_processor)} images...") 
    
    batch = processor(
        text=texts_for_processor, 
        images=images_for_processor, 
        return_tensors="pt", 
        padding=True,
        # max_length=args.max_length 
    )
    #print(f"Processor output keys: {batch.keys()}")
    # print(f"Input IDs shape: {batch['input_ids'].shape}, Attention Mask shape: {batch['attention_mask'].shape}") 
    if 'pixel_values' in batch:
        pass
        # print(f"Pixel Values shape: {batch['pixel_values'].shape}") 
    else:
        print("WARNING: 'pixel_values' not found in processor output. Image processing might have failed.")

    labels = batch["input_ids"].clone()

    image_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["boi_token"]
    )
    eoi_token_id = processor.tokenizer.convert_tokens_to_ids(
        processor.tokenizer.special_tokens_map["eoi_token"]
    )
    
    # print(f"Masking tokens: image_token_id={image_token_id}, eoi_token_id={eoi_token_id}, pad_token_id={processor.tokenizer.pad_token_id}") 

    special_token_ids_to_mask = {
        processor.tokenizer.pad_token_id,
        image_token_id,
        eoi_token_id,
    }
    special_token_ids_to_mask.add(262144) # Add other special tokens if necessary
    
    for token_id in special_token_ids_to_mask:
        if token_id is not None:
            labels[labels == token_id] = -100
            # print(f"Masked token ID {token_id} in labels.") 

    batch["labels"] = labels
    # print(f"Labels shape: {batch['labels'].shape}") 
    # print(f"--- Data Collator Batch End ---") 
    return batch

# --- 7. Defining Training Arguments ---
print(f"\n--- Defining Training Arguments ---")
num_train_epochs = 3
learning_rate = 2e-4
max_seq_length = 512

args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=learning_rate,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    push_to_hub=False,
    report_to="tensorboard",
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
    label_names=["labels"],
    max_length=max_seq_length,
)
print(f"SFTConfig arguments: {args}")


# --- 8. Custom callback function to save training metrics ---
class MetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file

        with open(self.log_file, "w") as f:
            f.write("Step,Epoch,Loss,Grad_Norm,Learning_Rate,Eval_Loss,Mean_Token_Accuracy\n")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:

            if "loss" in logs or "eval_loss" in logs:
                step = state.global_step
                epoch = logs.get("epoch", state.epoch) 
                
                loss = logs.get("loss", "N/A")
                grad_norm = logs.get("grad_norm", "N/A")
                learning_rate = logs.get("learning_rate", "N/A")
                
                eval_loss = logs.get("eval_loss", "N/A")
                mean_token_accuracy = logs.get("eval_mean_token_accuracy", "N/A") 
                

                log_entry = (
                    f"{step},"
                    f"{epoch:.2f}," 
                    f"{loss if isinstance(loss, (int, float)) else 'N/A'},"
                    f"{grad_norm if isinstance(grad_norm, (int, float)) else 'N/A'},"
                    f"{learning_rate if isinstance(learning_rate, (int, float)) else 'N/A'},"
                    f"{eval_loss if isinstance(eval_loss, (int, float)) else 'N/A'},"
                    f"{mean_token_accuracy if isinstance(mean_token_accuracy, (int, float)) else 'N/A'}\n"
                )
                
                with open(self.log_file, "a") as f:
                    f.write(log_entry)
                

                print(f"Logged Step: {step}, Epoch: {epoch:.2f}, Loss: {loss}, LR: {learning_rate}, Eval Loss: {eval_loss}, Acc: {mean_token_accuracy}")


# --- 9. Initializing SFTTrainer ---
print(f"\n--- Initializing SFTTrainer ---")
try:

    metrics_logger_callback = MetricsLoggerCallback(metrics_log_file)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"].shuffle(seed=42).select(range(min(100, len(data["validation"])))),
        peft_config=peft_config,
        data_collator=collate_fn,
        callbacks=[metrics_logger_callback], 
    )
    print(f"SFTTrainer initialized successfully.")

    print(f"\n--- Starting Training ---")
    trainer.train()
    print(f"\n--- Training Finished ---")

    print(f"Saving final model to {output_dir}/final_checkpoint")
    trainer.save_model(os.path.join(output_dir, "final_checkpoint"))
    processor.save_pretrained(os.path.join(output_dir, "final_checkpoint"))
    print(f"Model and processor saved.")

except Exception as e:
    print(f"ERROR: An error occurred during trainer initialization or training.")
    print(f"Error details: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
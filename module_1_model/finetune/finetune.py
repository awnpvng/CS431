import unsloth
import os
import sys
import json
import logging
import argparse
from typing import Optional
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq
from unsloth import FastLanguageModel, standardize_sharegpt, unsloth_train
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Config
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
LORA_R = 16
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]
LORA_ALPHA = 16
LORA_DROPOUT = 0 # Like dropout in neural network, but for LoRA it trains on LoRA weights which are very small compared to original model. So dropout is not really needed.
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # 1 GPU
GRADIENT_ACCUMULATION_STEPS = 16 # Accumulate gradients over 16 steps to simulate larger batch size
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"
WARMUP_STEPS = 0.1
WEIGHT_DECAY = 0.05 # Prevent overfitting by using L2 Regularization to penalize large weights after optimizer step
MAX_GRAD_NORM = 1.0 # Prevent exploding gradients

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "cot_data/cot_train.json")
VAL_DATA_PATH = os.path.join(SCRIPT_DIR, "cot_data/cot_val.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LoRA adapter on CoT dataset")

    # Add arguments for model name and output directory for Llama-3.2-3B-Instruct
    # parser.add_argument("--model_name", type=str, default="unsloth/Llama-3.2-3B-Instruct")
    # parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "Llama_3.2_3B_Instruct_adapter"))

    # Add arguments for model name and output directory for Qwen2.5-Coder-3B-Instruct
    # parser.add_argument("--model_name", type=str, default="unsloth/Qwen2.5-Coder-3B-Instruct")
    # parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "Qwen2.5_Coder_3B_Instruct_adapter"))

    # Add arguments for model name and output directory for Phi-3.5-mini-instruct-bnb-4bit
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit")
    parser.add_argument("--output_dir", type=str, default=os.path.join(SCRIPT_DIR, "Phi_4_mini_instruct_unsloth_bnb_4bit_adapter"))
    return parser.parse_args()

# Load and format dataset for input model
def load_dataset_from_json(train_path: str, val_path: str, tokenizer, model_name: str):
    """Load CoT JSON files and convert to HuggingFace Dataset."""
    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Standard ShareGPT
    train_dataset = Dataset.from_list([{"conversations": item["conversations"]} for item in train_data])
    val_dataset = Dataset.from_list([{"conversations": item["conversations"]} for item in val_data])

    # Apply ShareGPT standardization
    # Example for standarize_sharegpt
    # We now use `standardize_sharegpt` to convert ShareGPT style datasets into HuggingFace's generic format. This changes the dataset from looking like:
    # {"from": "system", "value": "You are an assistant"}
    # {"from": "human", "value": "What is 2+2?"}
    # {"from": "gpt", "value": "It's 4."}
    # to
    # {"role": "system", "content": "You are an assistant"}
    # {"role": "user", "content": "What is 2+2?"}
    # {"role": "assistant", "content": "It's 4."}

    train_dataset = standardize_sharegpt(train_dataset)
    val_dataset = standardize_sharegpt(val_dataset)

    # Use 'get_chat_template' to apply the right chat template to the tokenizer
    if model_name == "unsloth/Qwen2.5-Coder-3B-Instruct":
        tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    elif model_name == "unsloth/Llama-3.2-3B-Instruct":
        tokenizer = get_chat_template(tokenizer, chat_template="llama-3.2")
    elif model_name == "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit":
        tokenizer = get_chat_template(tokenizer, chat_template="phi-4")

    def format_dataset(dataset):
        def map_fn(examples):
            texts = [tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False) for conversation in examples["conversations"]]
            return {"text": texts}
        return dataset.map(map_fn, batched=True)

    train_dataset = format_dataset(train_dataset)
    val_dataset = format_dataset(val_dataset)

    return train_dataset, val_dataset

# Train function
def train():
    # Load arguments
    args = parse_args()

    # Load model (4-bit QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        full_finetuning=False
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    # Print trainable parameters
    model.print_trainable_parameters()

    # Load formatted dataset
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train data not found at {TRAIN_DATA_PATH}")
    elif not os.path.exists(VAL_DATA_PATH):
        raise FileNotFoundError(f"Validation data not found at {VAL_DATA_PATH}")

    train_dataset, val_dataset = load_dataset_from_json(TRAIN_DATA_PATH, VAL_DATA_PATH, tokenizer, args.model_name)

    # Training arguments
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=-1,
        max_grad_norm=MAX_GRAD_NORM,
        fp16=not model.config.dtype.__str__().__contains__("bfloat16"),
        bf16=model.config.dtype.__str__().__contains__("bfloat16"),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        seed=3407
    )

    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer), # tạo 1 lô dữ liệu từ train hoặc val dataset 
        dataset_text_field="text",
        dataset_num_proc=2, # quá trình chuẩn bị dữ liệu (2 luồng CPU, nhiều quá sẽ bị văng hoặc đơ màn hình)
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args
    )

    # Train
    trainer_stats = trainer.train() # Buggy gradient accumulation
    # trainer_stats = unsloth_train(trainer) # Use for llama3.2 to active gradient accumulation

    # Save LoRA adapter
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"LoRA adapter saved to: {args.output_dir}")

if __name__ == "__main__":
    train()

import os
import sys
import json
import logging
import argparse
import sqlite3
from tqdm import tqdm
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

# Add parent directory to sys.path to import evaluator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LoRA model on cot_test.json")
    parser.add_argument("--model_path", type=str, default="finetune/Phi_4_mini_instruct_unsloth_bnb_4bit_adapter")
    parser.add_argument("--test_data", type=str, default="finetune/cot_data/cot_test.json")
    parser.add_argument("--db_base_dir", type=str, default="spider_data/test_database")
    parser.add_argument("--output_file", type=str, default="evaluation/finetune_Phi_4_mini_instruct_unsloth_bnb_4bit.json")
    return parser.parse_args()

def extract_sql(response: str) -> str:
    sql = ""
    if "**SQL:**" in response:
        sql = response.split("**SQL:**")[-1].strip()
    else:
        # Fallback if the model didn't use the exact marker
        lines = response.split("\n")
        for i, line in enumerate(reversed(lines)):
            if "SELECT " in line.upper():
                sql = line.strip()
                break

    # Remove hallucinated generation continuations (e.g. "### Response:")
    if "###" in sql:
        sql = sql.split("###")[0].strip()
        
    # If the model hallucinates repeating statements separated by semicolon
    if ";" in sql:
        sql = sql.split(";")[0].strip()
        
    return sql

def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto"
    )

    with open(args.test_data, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    total_ex = 0
    total_esm = 0
    total_f1 = 0.0

    for idx, item in enumerate(tqdm(test_data)):
        # Construct inference prompt using ShareGPT conversation
        conversations = item.get("conversations", [])
        if not conversations or len(conversations) < 2:
            logger.warning(f"Invalid conversations format at item {idx}, skipping.")
            continue
            
        messages = [
            {"role": "system", "content": conversations[0]["value"]},
            {"role": "user", "content": conversations[1]["value"]}
        ]
        
        inference_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
            
        inputs = tokenizer([inference_prompt], return_tensors="pt").to("cuda")

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>")
        ]
        terminators = [t for t in terminators if t is not None]
        
        # Generate output
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=terminators
        )
        
        # Decode and extract new tokens only
        output_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        pred_sql = extract_sql(output_text)

        gold_sql = item["gold_sql"]
        db_id = item["db_id"]

        # Expected path format: spider_data/database/academic/academic.sqlite
        db_path = os.path.join(args.db_base_dir, db_id, f"{db_id}.sqlite")

        # Evaluate
        ex_score = Evaluator.evaluate_ex(pred_sql, gold_sql, db_path)
        esm_score = Evaluator.evaluate_esm(pred_sql, gold_sql)
        f1_score = Evaluator.evaluate_f1(pred_sql, gold_sql)

        total_ex += ex_score
        total_esm += esm_score
        total_f1 += f1_score

        results.append({
            "idx": idx,
            "db_id": db_id,
            "question": item.get("question", ""),
            "inference_prompt": inference_prompt,
            "generated_output": output_text,
            "pred_sql": pred_sql,
            "gold_sql": gold_sql,
            "EX": ex_score,
            "ESM": esm_score,
            "F1": f1_score
        })
        
        # Periodically log averages
        if (idx + 1) % 50 == 0:
            logger.info(f"Step {idx+1}/{len(test_data)} - Avg EX: {total_ex/(idx+1):.4f}, Avg ESM: {total_esm/(idx+1):.4f}, Avg F1: {total_f1/(idx+1):.4f}")

    avg_ex = total_ex / len(test_data)
    avg_esm = total_esm / len(test_data)
    avg_f1 = total_f1 / len(test_data)
    
    logger.info(f"Total Samples: {len(test_data)}")
    logger.info(f"Average EX: {avg_ex:.4f}")
    logger.info(f"Average ESM: {avg_esm:.4f}")
    logger.info(f"Average F1: {avg_f1:.4f}")

    # Save detailed results
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_ex": avg_ex,
                "avg_esm": avg_esm,
                "avg_f1": avg_f1,
                "total_samples": len(test_data)
            },
            "details": results
        }, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

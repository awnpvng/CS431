import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

class GroundedSchema:
    """
    Class chịu trách nhiệm xác định các Table và Column liên quan từ câu hỏi.
    """
    def extract(self, raw_output: str) -> str:
        if "**Grounded Schema:**" in raw_output:
            part = raw_output.split("**Grounded Schema:**")[-1]
            return part.split("**Logical Steps:**")[0].strip()
        return "N/A"

class IRSteps:
    """
    Class chịu trách nhiệm trích xuất các bước logic trung gian.
    """
    def extract(self, raw_output: str) -> str:
        if "**Logical Steps:**" in raw_output:
            part = raw_output.split("**Logical Steps:**")[-1]
            return part.split("**SQL:**")[0].strip()
        return "N/A"

class SQLGenerator:
    """
    Class chính điều phối model và xuất ra câu lệnh SQL cuối cùng.
    """
    def __init__(self, model_path: str):
        print(f"Loading Black Box Model từ: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Tự động nhận diện cấu hình máy
        if torch.cuda.is_available():
            print("🚀 Đã tìm thấy GPU NVIDIA! Đang load model ở chế độ 4-bit (Siêu tốc)...")
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True),
                device_map="auto"
            )
            self.device = "cuda"
        else:
            print("🐌 KHÔNG tìm thấy GPU NVIDIA! Đang load model lên CPU bằng RAM (Sẽ khá chậm và tốn RAM)...")
            self.model = AutoPeftModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu"
                # Đã bỏ load_in_4bit vì CPU không hỗ trợ tính năng này
            )
            self.device = "cpu"
            
        self.grounded_tool = GroundedSchema()
        self.ir_tool = IRSteps()
        
        self.system_prompt = (
            "You are an expert SQLite database engineer. "
            "Your task is to accurately translate a user's natural language question into a working SQL query using the provided database schema.\n"
            "Rules to follow:\n"
            "1. Use the 'Grounded Schema' to pinpoint exactly which tables and columns are relevant before logic planning.\n"
            "2. Think step-by-step to construct the SQL query logically.\n"
            "3. Output the final working SQLite query.\n\n"
        )

    def generate(self, question: str, schema_summary: str):
        # 1. Tạo Prompt theo đúng định dạng đã train
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Schema:\n{schema_summary}\n\nQuestion:\n{question}"}
        ]
        
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)
        
        # 2. Model Inference (Sinh ra toàn bộ CoT)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode kết quả
        full_response = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        # 3. Phân tách kết quả qua các class
        grounded = self.grounded_tool.extract(full_response)
        ir = self.ir_tool.extract(full_response)
        
        # Trích xuất SQL cuối cùng
        sql = "N/A"
        if "**SQL:**" in full_response:
            sql = full_response.split("**SQL:**")[-1].strip()
            # Clean SQL nếu có hallucination
            sql = sql.split(";")[0].strip()
        
        return {
            "grounded_schema": grounded,
            "ir_steps": ir,
            "sql": sql,
            "full_thought": full_response
        }


import sys
import os
import json
from dotenv import load_dotenv

# Thêm đường dẫn để Python tìm thấy các module của bạn
sys.path.append(os.path.join(os.getcwd(), "module_1_model"))
sys.path.append(os.path.join(os.getcwd(), "module_1_main"))

from module_1_model.database import Metadata
from module_1_main.sql_generator import SQLGenerator
from module_1_main.sql_executor import SQLExecutor, DatabaseInfor, DialectType

# --- HÀM TẠO SCHEMA SIÊU CHI TIẾT (RICH METADATA) ---
def get_rich_schema_summary(filepath="metadata.json"):
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    db_name = data.get("database_name", "ds2")
    summary = f"Database Schema for '{db_name}' with Rich Descriptions:\n\n"
    
    # 1. Đọc từng bảng và mô tả chi tiết
    tables = data.get("schemas", [])[0].get("tables", [])
    for table in tables:
        t_name = table["table_name"]
        t_desc = table.get("description", "")
        summary += f"Table [{t_name}]: {t_desc}\n"
        
        # Đọc các cột và mô tả, kiểu dữ liệu
        for col in table.get("columns", []):
            c_name = col["name"]
            c_type = col.get("type", "unknown")
            c_desc = col.get("description", "")
            summary += f"  - {c_name} ({c_type}): {c_desc}\n"
            
            # Gợi ý thêm các giá trị (ví dụ: F=Female, M=Male)
            if "value" in col:
                val_mapping = ", ".join([f"'{k}'={v}" for k, v in col["value"].items()])
                summary += f"      (Value Map: {val_mapping})\n"
                
        pks = table.get("primary_key", [])
        if pks:
            summary += f"  => Primary Key: {', '.join(pks)}\n"
            
        bls = table.get("business_logic", [])
        if bls:
            summary += f"  => Business Rules: {'; '.join(bls)}\n"
        summary += "\n"
        
    # 2. Đọc các mối quan hệ (Foreign Keys)
    relations = data.get("relations", [])
    if relations:
        summary += "Relationships (Foreign Keys):\n"
        for rel in relations:
            t1, c1 = rel["source_table"], rel["source_column"]
            t2, c2 = rel["target_table"], rel["target_column"]
            summary += f"- {t1}.{c1} = {t2}.{c2}\n"
            
    return summary, db_name

# --- BƯỚC 1: KHỞI TẠO CÁC THÀNH PHẦN ---
# 1.1 Khởi tạo Schema với Rich Metadata từ ds2_full.json
schema_summary, db_id = get_rich_schema_summary("metadata.json")

# 1.2 Khởi tạo SQL Generator (Black Box)
MODEL_PATH = os.path.join(os.getcwd(), "module_1_model", "finetune", "Qwen2.5_Coder_3B_Instruct_adapter")
generator = SQLGenerator(MODEL_PATH)

# 1.3 Cấu hình Database và Khởi tạo Executor (Đổi sang SQLite)
db_infor = DatabaseInfor(
    dialect_type=DialectType.SQLITE,
    config={
        "database_path": "ds2.db"
    }
)
conn_str = db_infor.get_connection_string
executor = SQLExecutor(connection_string=conn_str)

# --- BƯỚC 2: QUY TRÌNH CHẠY HỆ THỐNG ---
def run_text_to_sql_system(question, schema_text):
    print(f"\n--- Đang xử lý câu hỏi ---")
    
    # Kết nối Database
    if not executor.connect_database():
        print("Lỗi: Không thể kết nối đến Database!")
        return None, None

    try:
        # Sinh SQL (Qua 3 bước trong Black Box: Grounded -> IR -> SQL)
        gen_result = generator.generate(question, schema_text)
        sql_query = gen_result['sql']
        
        print(f"SQL được tạo (Chuẩn SQLite):\n{sql_query}")
        
        # Thực thi SQL để lấy DataFrame kết quả
        df_result = executor.execute_query(sql_query)
        
        return gen_result, df_result
    
    finally:
        # Luôn đóng kết nối sau khi xong
        executor.close_connection()

# --- CHẠY THỬ ---
if __name__ == "__main__":
    print(f"Đã nạp thành công schema cho database: {db_id}")
    
    # Câu hỏi bạn muốn test
    user_question = "Who is the top-grossing actor in the 'Comedy' category?"
    # user_question = "tổng số lượng đơn hàng"
    steps, dataframe = run_text_to_sql_system(user_question, schema_summary)
    
    if dataframe is not None:
        print("\n--- KẾT QUẢ PHÂN TÍCH (COT) ---")
        print(f"Grounded Schema: {steps['grounded_schema']}")
        print(f"Logical Steps: {steps['ir_steps']}")
        
        print("\n--- DỮ LIỆU XUẤT RA (DATAFRAME) ---")
        print(dataframe)

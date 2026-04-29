import sys
import os
import json
from dotenv import load_dotenv

# Thêm đường dẫn để Python tìm thấy các module của bạn
sys.path.append(os.path.join(os.getcwd(), "module_1"))
sys.path.append(os.path.join(os.getcwd(), "module_2"))

from module_2.sql_generator import SQLGenerator
from module_2.sql_executor import SQLExecutor, DatabaseInfor, DialectType

# --- BƯỚC 0: LOAD CẤU HÌNH TỪ .ENV ---
load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT"))
POSTGRES_DB = os.getenv("POSTGRES_DB")

# --- HÀM ĐỌC SCHEMA TỪ FILE DS2 (CUSTOM FORMAT) ---
def get_schema_summary_from_ds2(filepath="ds2.json"):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    db_name = data.get("database_name", "ds2")
    summary = f"Schema for database '{db_name}':\n"
    
    # Đọc các bảng và cột
    tables = data.get("schemas", [])[0].get("tables", [])
    for table in tables:
        t_name = table["table_name"]
        cols = [col["name"] for col in table.get("columns", [])]
        summary += f"Table {t_name}: {', '.join(cols)}\n"
        
    # Đọc các relationships
    relations = data.get("relations", [])
    if relations:
        summary += "\nRelationships:\n"
        for rel in relations:
            t1, c1 = rel["source_table"], rel["source_column"]
            t2, c2 = rel["target_table"], rel["target_column"]
            summary += f"- {t1}.{c1} = {t2}.{c2}\n"
            
    return summary, db_name

# --- BƯỚC 1: KHỞI TẠO CÁC THÀNH PHẦN ---
# 1.1 Lấy Schema tóm tắt từ ds2.json
schema_summary, db_id = get_schema_summary_from_ds2("ds2.json")

# 1.2 Khởi tạo SQL Generator (Black Box)
MODEL_PATH = r"D:\code_nam_3\deep learning\final_project\module_1\finetune\Qwen2.5_Coder_3B_Instruct_adapter"
generator = SQLGenerator(MODEL_PATH)

# 1.3 Cấu hình Database và Khởi tạo Executor
db_infor = DatabaseInfor(
    dialect_type=DialectType.POSTGRESQL,
    config={
        "host": POSTGRES_HOST,
        "port": POSTGRES_PORT,
        "user": POSTGRES_USER,
        "password": POSTGRES_PASSWORD,
        "database": POSTGRES_DB
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
        
        print(f"SQL được tạo:\n{sql_query}")
        
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
    user_question = "tính số lượng đơn hàng"
    
    steps, dataframe = run_text_to_sql_system(user_question, schema_summary)
    
    if dataframe is not None:
        print("\n--- KẾT QUẢ PHÂN TÍCH (COT) ---")
        print(f"Grounded Schema: {steps['grounded_schema']}")
        print(f"Logical Steps: {steps['ir_steps']}")
        
        print("\n--- DỮ LIỆU XUẤT RA (DATAFRAME) ---")
        print(dataframe)

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sql_executor import SQLExecutor, DatabaseInfor, DialectType
import os
from dotenv import load_dotenv

load_dotenv()
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT"))
POSTGRES_DB = os.getenv("POSTGRES_DB")

MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_PORT = int(os.getenv("MYSQL_PORT"))
MYSQL_DB = os.getenv("MYSQL_DB")

if __name__ == "__main__":
    # db_infor = DatabaseInfor(
    #     dialect_type=DialectType.MYSQL,
    #     config={
    #         "host": MYSQL_HOST,
    #         "port": MYSQL_PORT,
    #         "user": MYSQL_USER,
    #         "password": MYSQL_PASSWORD,
    #         "database": MYSQL_DB
    #     }
    # )
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
    conn_str = db_infor.get_connection_string #tạo url database

    executor = SQLExecutor(connection_string=conn_str)
    
    if executor.connect_database():
        print("connect success!\n")
        
        test_query = """
SELECT COUNT("public"."orderlines"."orderid") AS "so_luong_don_hang" FROM "public"."orderlines" WHERE EXTRACT(MONTH FROM "public"."orderlines"."orderdate") = 1
        """
        
        print("khởi tạo kết quả:")
        result = executor.execute_query(test_query)
        print(result)
        
        executor.close_connection()
        print("\nclose connection")
    else:
        print("failed")

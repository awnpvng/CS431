import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from typing import Union
import traceback

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json



class DialectType(Enum):
    MYSQL      = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"



config_requirement= {
    DialectType.POSTGRESQL: ["host", "port", "user", "password", "database"],
    DialectType.MYSQL:      ["host", "port", "user", "password", "database"],
    DialectType.SQLITE:     ["database_path"]
}
@dataclass
class DatabaseInfor:
    dialect_type: DialectType
    config: Dict[str, Any]  

    def __post_init__(self): # Hàm tự động: kiểm tra cấu hình ngay khi vừa khởi tạo Object.
        requirement = config_requirement.get(self.dialect_type, [])
        missing_key = [key for key in requirement if key not in self.config]
        if missing_key:
            raise ValueError(
                f"missing config for {self.dialect_type.value.upper()}: "
                f"{', '.join(missing_key)}"
            )
    @property
    def get_connection_string(self) -> str:
        c = self.config 
        
        if self.dialect_type == DialectType.POSTGRESQL:
            return f"postgresql+psycopg2://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"        
        elif self.dialect_type == DialectType.SQLITE:
            # SQLite chỉ cần đường dẫn file
            return f"sqlite:///{c['database_path']}"
        elif self.dialect_type == DialectType.MYSQL:
            return f"mysql+pymysql://{c['user']}:{c['password']}@{c['host']}:{c['port']}/{c['database']}"
        raise NotImplementedError(f"Not implement URL for {self.dialect_type}")
        
class SQLExecutor:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.engine = None
        self.conn = None
        
    def connect_database(self):
        success = False
        try:
            self.engine = create_engine(self.connection_string)
            self.conn = self.engine.connect()
            success = True
        except Exception as e:
            traceback.print_exc()
            print(f"Connection failed: {e}")
        return success
    
    def execute_query(self, query: str) -> Union[pd.DataFrame, str]:
        from sqlalchemy import text
        result = "Not connected to database"
        if self.conn:
            try:
                result = pd.read_sql(text(query), self.conn)
            except Exception as e:
                result = f"[SQL ERROR] {e}"
        # if isinstance(result, pd.DataFrame):
        #     result = self.validate_df(result)
        return result

    # @staticmethod
    # def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    #     rename_map = {}
    #     divisor_map = {}

    #     for col in df.columns:
    #         if not pd.api.types.is_numeric_dtype(df[col]):
    #             continue
    #         max_val = df[col].abs().max()
    #         if pd.isna(max_val):
    #             continue

    #         if max_val >= 1_000_000_000:
    #             divisor_map[col] = 1_000_000_000
    #             rename_map[col] = f"{col} [đơn vị tỷ]"
    #         elif max_val >= 1_000_000:
    #             divisor_map[col] = 1_000_000
    #             rename_map[col] = f"{col} [đơn vị triệu]"
    #         elif max_val > 10_000:
    #             divisor_map[col] = 10_000
    #             rename_map[col] = f"{col} [đơn vị vạn]"

    #     # Chia giá trị cho đơn vị tương ứng
    #     for col, divisor in divisor_map.items():
    #         df[col] = (df[col] / divisor).round(2)

    #     # Đổi tên cột sau khi đã chia
    #     if rename_map:
    #         df = df.rename(columns=rename_map)
    #     return df

    def close_connection(self):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.engine:
            self.engine.dispose()
            self.engine = None




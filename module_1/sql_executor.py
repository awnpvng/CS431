import pandas as pd
from database import Database
from typing import Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SQLExecutor:
    """
    Responsible for running the SQL query against the database
    and getting the resulting dataframe.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def execute(self, generated_sql_data: dict) -> pd.DataFrame:
        """
        Executes the SQL query and returns the Pandas DataFrame.
        """
        sql_query = generated_sql_data["sql"]
        logger.info(f"Executing query: {sql_query}")
        
        df = self.db.execute_query(sql_query)
        return df
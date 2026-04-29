import os
import json
import sqlite3
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Metadata:
    """
    Represents the Metadata store containing:
    - dataset schema
    - relationships
    - business analysis
    """
    
    def __init__(self, tables_path: str = None) -> None:
        self.schemas = {}
        if tables_path and os.path.exists(tables_path):
            with open(tables_path, 'r', encoding='utf-8') as f:
                tables_data = json.load(f)

            for database in tables_data:
                self.schemas[database['db_id']] = database

    def get_schema_summary(self, db_id: str = None) -> str:
        """Returns a string representation of the schema for LLM."""

        # Check if schemas are loaded and db_id exists
        if not self.schemas or not db_id or db_id not in self.schemas:
            logger.error(f"Schema not found for db_id: {db_id}")
            raise ValueError(f"Schema not found for db_id: {db_id}")

        # Get database from db_id
        database = self.schemas[db_id]

        # Initialize summary string
        summary = f"Schema for database '{db_id}':\n"

        # Get table names
        table_names = database.get("table_names_original", [])

        # Get column names
        column_names = database.get("column_names_original", [])

        # Create tables dictionary
        tables = {i: [] for i in range(len(table_names))}
        
        # Populate tables dictionary with columns
        for col in column_names:
            table_idx, col_name = col
            if table_idx >= 0:
                tables[table_idx].append(col_name)

        # Add table names and columns to summary
        for table_idx, table_name in enumerate(table_names):
            cols = ", ".join(tables[table_idx])
            
            # Update summary
            summary += f"Table {table_name}: {cols}\n"
            
        # If foreign key exists
        if database.get("foreign_keys"):
            summary += "\nRelationships:\n"
            for fk in database["foreign_keys"]:
                fk_col1 = column_names[fk[0]]
                fk_col2 = column_names[fk[1]]

                # Get table and column names from foreign key
                table1 = table_names[fk_col1[0]]
                col1 = fk_col1[1]

                table2 = table_names[fk_col2[0]]
                col2 = fk_col2[1]

                # Update summary
                summary += f"- {table1}.{col1} = {table2}.{col2}\n"

        return summary


class Database:
    """
    Represents the DBMS (use SQLite3)
    """

    def __init__(self, db_path: str = None) -> None:
        self.conn = None
        if db_path:
            if os.path.exists(db_path):
                logger.info(f"Connecting to database at: {db_path}")
                self.conn = sqlite3.connect(db_path)
            else:
                logger.error(f"Database SQLite file not found at: {db_path}. Please provide correct Spider database paths.")
                raise FileNotFoundError(f"Database SQLite file not found at: {db_path}. Please provide correct Spider database paths.")

    def execute_query(self, query: str) -> pd.DataFrame:
        """Executes a SQL query from user and returns a pandas DataFrame."""
        try:
            logger.info(f"Executing SQL query: {query}")
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return pd.DataFrame({"Error": [str(e)]})

    def disconnect(self) -> None:
        """Disconnect from a Database."""
        logger.info(f"Disconnecting from database")
        self.conn.close()
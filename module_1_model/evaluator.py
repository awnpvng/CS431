import sqlite3
import pandas as pd
import re
import logging
from typing import Set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Evaluator:
    @staticmethod
    def evaluate_ex(pred_sql: str, gold_sql: str, db_path: str) -> int:
        """
        Execution Accuracy (EX):
        Executes both SQL queries on the actual database.
        Return 1 if the resulting datasets have the same rows (order-agnostic), 0 otherwise.
        """

        if not pred_sql or not gold_sql or not db_path:
            return 0

        try:
            conn = sqlite3.connect(db_path)
            gold_df = pd.read_sql_query(gold_sql, conn)
            pred_df = pd.read_sql_query(pred_sql, conn)
            conn.close()

            # Compare dataframes
            if gold_df.shape != pred_df.shape:
                return 0

            # Use set to compare rows
            gold_set = set(tuple(x) for x in gold_df.to_numpy())
            pred_set = set(tuple(x) for x in pred_df.to_numpy())

            return 1 if gold_set == pred_set else 0

        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            return 0

    @staticmethod
    def evaluate_esm(pred_sql: str, gold_sql: str) -> int:
        """
        Exact Set Match (ESM):
        Heuristically compares the two queries by standardizing spaces, lowercasing,
        and stripping common LLM additions (AS aliases, specific COUNTs).
        Returns 1 if they match exactly, 0 otherwise.
        """

        if not pred_sql or not gold_sql:
            return 0

        def normalize_sql(sql: str) -> str:
            # Lowercase and remove semicolons
            sql = sql.lower().replace(";", "").strip()
            # Remove 'AS alias' artifacts
            sql = re.sub(r'\s+as\s+\w+', '', sql)
            # Standarize COUNT(...) to COUNT(*)
            sql = re.sub(r'count\s\(\s*(distinct\s+)?[\w\.]+\s*\)', 'count(*)', sql)
            # Remove optional table prefixes (e.g., singer.Name -> Name)
            sql = re.sub(r'\b[a-z_][a-z0-9_]*\.', '', sql)
            # Standarize whitespace and commas
            sql = sql.replace(',', ' , ')
            return " ".join(sql.split())

        norm_pred = normalize_sql(pred_sql)
        norm_gold = normalize_sql(gold_sql)
        return 1 if norm_pred == norm_gold else 0

    @staticmethod
    def evaluate_f1(pred_sql: str, gold_sql: str) -> float:
        """
        Component Match (F1-Score)
        Extract alphanumeric tokens (keywords, tables, columns).
        Return the F1-Score showing the overlap between predicted and gold tokens.
        """

        if not pred_sql or not gold_sql:
            return 0.0

        def get_tokens(sql: str) -> Set[str]:
            sql = sql.lower().replace(';', '')
            words = set(re.findall(r'\b\w+\b', sql))
            return words

        pred_tokens = get_tokens(pred_sql)
        gold_tokens = get_tokens(gold_sql)

        if not pred_tokens or not gold_tokens:
            return 0.0

        intersection = pred_tokens.intersection(gold_tokens)
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(gold_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
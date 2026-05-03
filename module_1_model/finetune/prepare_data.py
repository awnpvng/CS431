import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database import Metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# AGG functions
AGG_OPS = {0: "", 1: "MAX", 2: "MIN", 3: "COUNT", 4: "SUM", 5: "AVG"}

# Another operations
WHERE_OPS = {
    0: "NOT IN",
    1: "BETWEEN",
    2: "=",
    3: ">",
    4: "<",
    5: ">=",
    6: "<=",
    7: "!=",
    8: "IN",
    9: "LIKE",
    10: "IS",
    11: "EXISTS",
}
ORDER_DIR = {1: "ASC", 2: "DESC"}

# Helper: resolve column index → (table_name, col_name)
def resolve_col(col_idx: int, schema: Dict) -> Tuple[str, str]:
    """Return (table_name, column_name) for a column index."""
    col_names = schema.get("column_names_original", [])
    table_names = schema.get("table_names_original", [])

    if col_idx < 0 or col_idx >= len(col_names):
        return ("", "*")

    table_idx, col_name = col_names[col_idx]
    if table_idx < 0:
        return ("", col_name)  # wildcard *
    table_name = table_names[table_idx] if table_idx < len(table_names) else ""
    return (table_name, col_name)

# Format column with optional agg
def fmt_col(col_idx: int, schema: Dict, agg: int = 0) -> str:
    """Format column with optional aggregate function."""
    table, col = resolve_col(col_idx, schema)
    qualified = f"{table}.{col}" if table else col
    if agg and agg in AGG_OPS and AGG_OPS[agg]:
        return f"{AGG_OPS[agg]}({qualified})"
    return qualified


# Extract Grounded Schema from sql field
def extract_grounded_schema(sql: Dict, schema: Dict) -> str:
    """
    Extract the subset of tables + columns actually used in the gold SQL.
    Returns a human-readable string.
    """
    table_names = schema.get("table_names_original", [])
    col_names = schema.get("column_names_original", [])
    column_names_lookup = schema.get("column_names_original", [])
    foreign_keys = schema.get("foreign_keys", [])

    # Tables used
    used_table_indices = set()
    for unit_type, unit_val in sql.get("from", {}).get("table_units", []):
        if unit_type == "table_unit":
            used_table_indices.add(unit_val)

    # Columns used
    used_col_indices = set()

    def collect_col_unit(col_unit):
        if isinstance(col_unit, list) and len(col_unit) >= 2:
            inner = col_unit[1]
            if isinstance(inner, list) and len(inner) >= 2:
                used_col_indices.add(inner[1])

    def collect_val_unit(val_unit):
        if isinstance(val_unit, list) and len(val_unit) >= 2:
            collect_col_unit(val_unit[1])
            if val_unit[0] != 0 and len(val_unit) > 2:
                collect_col_unit(val_unit[2])

    # SELECT columns
    select_distinct, select_items = sql.get("select", [False, []])
    for agg_id, val_unit in select_items:
        collect_val_unit(val_unit)

    # WHERE columns
    for cond in sql.get("where", []):
        if isinstance(cond, list) and len(cond) >= 3:
            collect_val_unit(cond[2])

    # GROUP BY
    for col_unit in sql.get("groupBy", []):
        if isinstance(col_unit, list) and len(col_unit) >= 2:
            used_col_indices.add(col_unit[1])

    # ORDER BY
    order = sql.get("orderBy", [])
    if order and len(order) >= 2:
        for val_unit in order[1]:
            collect_val_unit(val_unit)

    # HAVING
    for cond in sql.get("having", []):
        if isinstance(cond, list) and len(cond) >= 3:
            collect_val_unit(cond[2])

    # JOIN conditions from "from.conds"
    for cond in sql.get("from", {}).get("conds", []):
        if isinstance(cond, list) and len(cond) >= 3:
            collect_val_unit(cond[2])
            if len(cond) > 3 and isinstance(cond[3], list):
                used_col_indices.add(cond[3][1] if isinstance(cond[3], list) else cond[3])

    # Build used_table_indices from column ownership
    for col_idx in used_col_indices:
        if 0 <= col_idx < len(col_names):
            t_idx = col_names[col_idx][0]
            if t_idx >= 0:
                used_table_indices.add(t_idx)

    # Format output
    lines = []

    # Tables section
    used_tables = [table_names[i] for i in sorted(used_table_indices) if i < len(table_names)]
    lines.append(f"Tables: {', '.join(used_tables) if used_tables else 'N/A'}")

    # Columns per table
    for t_idx in sorted(used_table_indices):
        if t_idx >= len(table_names):
            continue
        tname = table_names[t_idx]
        cols = [
            col_names[c][1]
            for c in sorted(used_col_indices)
            if 0 <= c < len(col_names) and col_names[c][0] == t_idx
        ]
        if cols:
            lines.append(f"  {tname}: {', '.join(cols)}")

    # Joins from foreign keys (only between used tables)
    joins = []
    for fk in foreign_keys:
        if len(fk) < 2:
            continue
        c1_idx, c2_idx = fk[0], fk[1]
        if c1_idx >= len(col_names) or c2_idx >= len(col_names):
            continue
        t1 = col_names[c1_idx][0]
        t2 = col_names[c2_idx][0]
        if t1 in used_table_indices and t2 in used_table_indices:
            t1n = table_names[t1] if t1 < len(table_names) else "?"
            t2n = table_names[t2] if t2 < len(table_names) else "?"
            c1n = col_names[c1_idx][1]
            c2n = col_names[c2_idx][1]
            joins.append(f"{t1n}.{c1n} = {t2n}.{c2n}")
    if joins:
        lines.append(f"Joins: {'; '.join(joins)}")

    return "\n".join(lines)


# Generate IR Steps from sql field
def generate_ir_steps(sql: Dict, schema: Dict) -> str:
    """
    Generate human-readable logical steps from the parsed SQL structure.
    """
    steps = []
    step_num = 1

    # ── FROM / JOIN ──
    table_units = sql.get("from", {}).get("table_units", [])
    from_conds = sql.get("from", {}).get("conds", [])
    table_names = schema.get("table_names_original", [])

    from_tables = []
    for unit_type, unit_val in table_units:
        if unit_type == "table_unit" and unit_val < len(table_names):
            from_tables.append(table_names[unit_val])

    if len(from_tables) == 1:
        steps.append(f"{step_num}. Access table: {from_tables[0]}")
    elif len(from_tables) > 1:
        join_parts = []
        for cond in from_conds:
            if isinstance(cond, list) and len(cond) >= 4:
                try:
                    left = fmt_col(cond[2][1][1], schema) if isinstance(cond[2], list) else "?"
                    right = fmt_col(cond[3][1], schema) if isinstance(cond[3], list) else "?"
                    join_parts.append(f"{left} = {right}")
                except Exception:
                    pass
        join_str = f" ON {'; '.join(join_parts)}" if join_parts else ""
        steps.append(f"{step_num}. JOIN {' ↔ '.join(from_tables)}{join_str}")
    step_num += 1

    # ── WHERE ──
    where_conds = sql.get("where", [])
    where_parts = []
    for cond in where_conds:
        if not isinstance(cond, list):
            continue  # "AND"/"OR" strings
        if len(cond) < 4:
            continue
        not_op, op_id, val_unit, val1, val2 = cond[0], cond[1], cond[2], cond[3], cond[4]
        try:
            col_str = fmt_col(val_unit[1][1], schema)
        except Exception:
            col_str = "?"
        op_str = WHERE_OPS.get(op_id, "?")
        not_str = "NOT " if not_op else ""
        val_str = str(val1) if val1 is not None else ""
        if val2 is not None:
            val_str += f" AND {val2}"
        where_parts.append(f"{not_str}{col_str} {op_str} {val_str}".strip())

    if where_parts:
        steps.append(f"{step_num}. Filter where: {'; '.join(where_parts)}")
        step_num += 1

    # ── GROUP BY ──
    group_by = sql.get("groupBy", [])
    if group_by:
        gb_cols = []
        for col_unit in group_by:
            if isinstance(col_unit, list) and len(col_unit) >= 2:
                try:
                    gb_cols.append(fmt_col(col_unit[1], schema))
                except Exception:
                    pass
        if gb_cols:
            steps.append(f"{step_num}. Group by: {', '.join(gb_cols)}")
            step_num += 1

    # ── HAVING ──
    having_conds = sql.get("having", [])
    having_parts = []
    for cond in having_conds:
        if not isinstance(cond, list) or len(cond) < 4:
            continue
        not_op, op_id, val_unit, val1, val2 = cond[0], cond[1], cond[2], cond[3], cond[4]
        try:
            agg = val_unit[0] if val_unit[0] else 0
            col_str = fmt_col(val_unit[1][1], schema, agg)
        except Exception:
            col_str = "?"
        op_str = WHERE_OPS.get(op_id, "?")
        not_str = "NOT " if not_op else ""
        having_parts.append(f"{not_str}{col_str} {op_str} {val1}")
    if having_parts:
        steps.append(f"{step_num}. Having: {'; '.join(having_parts)}")
        step_num += 1

    # ── ORDER BY ──
    order = sql.get("orderBy", [])
    if order and len(order) >= 2:
        direction = ORDER_DIR.get(order[0], "")
        ord_cols = []
        for val_unit in order[1]:
            try:
                agg = val_unit[0] if val_unit[0] else 0
                ord_cols.append(fmt_col(val_unit[1][1], schema, agg))
            except Exception:
                pass
        if ord_cols:
            steps.append(f"{step_num}. Order by: {', '.join(ord_cols)} {direction}".strip())
            step_num += 1

    # ── LIMIT ──
    limit = sql.get("limit")
    if limit is not None:
        steps.append(f"{step_num}. Limit results to: {limit}")
        step_num += 1

    # ── SELECT ──
    select_distinct, select_items = sql.get("select", [False, []])
    sel_cols = []
    for agg_id, val_unit in select_items:
        try:
            col_str = fmt_col(val_unit[1][1], schema, agg_id)
            sel_cols.append(col_str)
        except Exception:
            pass
    dist_str = "DISTINCT " if select_distinct else ""
    if sel_cols:
        steps.append(f"{step_num}. Select {dist_str}{', '.join(sel_cols)}")

    return "\n".join(steps) if steps else "1. Select all rows from the table"


# Build prompt string
SYSTEM_MSG = (
    "You are an expert SQLite database engineer. "
    "Your task is to accurately translate a user's natural language question into a working SQL query using the provided database schema.\n"
    "Rules to follow:\n"
    "1. Use the 'Grounded Schema' to pinpoint exactly which tables and columns are relevant before logic planning.\n"
    "2. Formulate explicitly clear 'Logical Steps' mapping the problem to SQL mechanics (e.g., JOINs, GROUP BY).\n"
    "3. Do NOT hallucinate column names that are not in the schema. Avoid unnecessary complex UNIONs unless strictly required.\n"
    "4. Output the final working SQLite query."
)

def build_sharegpt_conversations(question: str, full_schema: str, grounded: str, ir_steps: str, sql: str) -> List[Dict[str, str]]:
    return [
        {"from": "system", "value": SYSTEM_MSG},
        {"from": "human", "value": f"Schema:\n{full_schema}\n\nQuestion:\n{question}"},
        {"from": "gpt", "value": f"**Grounded Schema:**\n{grounded}\n\n**Logical Steps:**\n{ir_steps}\n\n**SQL:**\n{sql}"}
    ]


# Process one Spider item
def process_item(item: Dict, metadata: Metadata, tables_raw: Dict) -> Optional[Dict]:
    question = item.get("question", "").strip()
    db_id = item.get("db_id", "")
    gold_sql = item.get("query", "").strip()
    sql_struct = item.get("sql", {})

    if not question or not gold_sql or not db_id:
        return None

    schema = tables_raw.get(db_id)
    if schema is None:
        logger.warning(f"Schema not found for db_id={db_id}, skipping.")
        return None

    try:
        full_schema = metadata.get_schema_summary(db_id)
    except Exception as e:
        logger.warning(f"Schema summary error for {db_id}: {e}")
        return None

    try:
        grounded = extract_grounded_schema(sql_struct, schema)
        ir_steps = generate_ir_steps(sql_struct, schema)
    except Exception as e:
        logger.warning(f"CoT extraction error for '{question}': {e}")
        grounded = "N/A"
        ir_steps = "1. Execute query directly"

    conversations = build_sharegpt_conversations(question, full_schema, grounded, ir_steps, gold_sql)

    return {
        "db_id": db_id,
        "question": question,
        "gold_sql": gold_sql,
        "grounded_schema": grounded,
        "ir_steps": ir_steps,
        "conversations": conversations,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare CoT dataset from Spider")
    parser.add_argument("--data_dir", default="spider_data", help="Spider data directory")
    parser.add_argument("--output_dir", default="finetune/cot_data", help="Output directory")
    parser.add_argument("--include_others", action="store_true", default=True,
                        help="Also include train_others.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tables (tables.json and test_tables.json)
    # Train
    tables_train_path = os.path.join(args.data_dir, "tables.json")
    logger.info(f"Loading tables from {tables_train_path}")

    with open(tables_train_path, "r", encoding="utf-8") as f:
        tables_train_list = json.load(f)
    
    tables_train_raw = {table["db_id"]: table for table in tables_train_list}
    train_metadata = Metadata(tables_path=tables_train_path)
    logger.info(f"Loaded {len(tables_train_raw)} database schemas")

    # Test
    tables_test_path = os.path.join(args.data_dir, "test_tables.json")
    logger.info(f"Loading test tables from {tables_test_path}")

    with open(tables_test_path, "r", encoding="utf-8") as f:
        tables_test_list = json.load(f)

    tables_test_raw = {table["db_id"]: table for table in tables_test_list}
    test_metadata =  Metadata(tables_path=tables_test_path)
    logger.info(f"Loaded {len(tables_test_raw)} database schemas")

    # Load data
    # Train
    train_files = [os.path.join(args.data_dir, "train_spider.json")]
    if args.include_others:
        others_path = os.path.join(args.data_dir, "train_others.json")
        if os.path.exists(others_path):
            train_files.append(others_path)

    train_data = []
    for fpath in train_files:
        logger.info(f"Loading {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            train_data.extend(json.load(f))

    # Val
    val_path = os.path.join(args.data_dir, "dev.json")
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    logger.info(f"Total val samples: {len(val_data)}")

    # Test
    test_path = os.path.join(args.data_dir, "test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"Total test samples: {len(test_data)}")

    # Process
    def process(data: List[Dict], split_name: str, metadata: Metadata, tables: Dict) -> List[Dict]:
        # Store result
        results = []
        # Data error
        skipped = 0
        for i, item in enumerate(data):
            processed = process_item(item, metadata, tables)
            if processed:
                results.append(processed)
            else:
                skipped += 1
        logger.info(f"[{split_name}] Done: {len(results)} samples, {skipped} skipped")
        return results

    train_results = process(train_data, "train", train_metadata, tables_train_raw)
    val_results = process(val_data, "val", train_metadata, tables_train_raw)
    test_results = process(test_data, "test", test_metadata, tables_test_raw)

    # Save
    def save(path: str, data: List[Dict]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} samples to {path}")

    train_out = os.path.join(args.output_dir, "cot_train.json")
    val_out = os.path.join(args.output_dir, "cot_val.json")
    test_out = os.path.join(args.output_dir, "cot_test.json")

    save(train_out, train_results)
    save(val_out, val_results)
    save(test_out, test_results)

if __name__ == "__main__":
    main()

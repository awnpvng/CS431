"""
EDA script for cot_train.json
Phân tích:
1. Phân bố độ dài câu hỏi (Question Length Distribution)
2. Từ vựng và N-grams phổ biến (Vocabulary & N-grams)
3. Số lượng component trong mỗi câu SQL
4. Tần suất xuất hiện của các mệnh đề SQL (SQL Clause Frequencies)
5. Độ sâu Subqueries và Số lượng bảng được JOIN
"""

import json
import re
import os
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "cot_train.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "eda_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[*] Loading data from {DATA_PATH} ...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [d["question"] for d in data]
sql_queries = [d["gold_sql"] for d in data]
N = len(data)
print(f"[*] Total samples: {N}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. PHÂN BỐ ĐỘ DÀI CÂU HỎI (Question Length Distribution)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. QUESTION LENGTH DISTRIBUTION")
print("=" * 70)

q_word_lens = [len(q.split()) for q in questions]
q_char_lens = [len(q) for q in questions]

print(f"  Word-level  => min={min(q_word_lens)}, max={max(q_word_lens)}, "
      f"mean={np.mean(q_word_lens):.2f}, median={np.median(q_word_lens):.1f}, "
      f"std={np.std(q_word_lens):.2f}")
print(f"  Char-level  => min={min(q_char_lens)}, max={max(q_char_lens)}, "
      f"mean={np.mean(q_char_lens):.2f}, median={np.median(q_char_lens):.1f}")

# Percentiles
for p in [25, 50, 75, 90, 95, 99]:
    print(f"    P{p}: {np.percentile(q_word_lens, p):.0f} words / "
          f"{np.percentile(q_char_lens, p):.0f} chars")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(q_word_lens, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(np.mean(q_word_lens), color="red", linestyle="--", label=f"Mean={np.mean(q_word_lens):.1f}")
axes[0].axvline(np.median(q_word_lens), color="orange", linestyle="--", label=f"Median={np.median(q_word_lens):.1f}")
axes[0].set_title("Question Length (words)")
axes[0].set_xlabel("Number of words")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].hist(q_char_lens, bins=50, color="darkcyan", edgecolor="white", alpha=0.85)
axes[1].axvline(np.mean(q_char_lens), color="red", linestyle="--", label=f"Mean={np.mean(q_char_lens):.1f}")
axes[1].set_title("Question Length (characters)")
axes[1].set_xlabel("Number of characters")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "01_question_length_distribution.png"), dpi=150)
plt.close()
print(f"  => Saved: 01_question_length_distribution.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TỪ VỰNG VÀ N-GRAMS PHỔ BIẾN (Vocabulary & N-grams)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("2. VOCABULARY & N-GRAMS")
print("=" * 70)

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in", "for", "to",
    "and", "or", "that", "this", "it", "with", "on", "at", "by", "from",
    "as", "be", "has", "have", "had", "do", "does", "did", "but", "not",
    "so", "if", "than", "too", "very", "can", "will", "just", "should",
    "its", "his", "her", "their", "my", "your", "our", "me", "him",
    "them", "we", "they", "i", "you", "he", "she", "who", "which",
    "what", "where", "when", "how", "there", "each", "all", "both",
    "few", "more", "most", "other", "some", "such", "no", "nor",
    "only", "own", "same", "about", "up", "out", "also", "been", "being",
}


def tokenize(text):
    return re.findall(r"[a-zA-Z]+", text.lower())


def get_ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


all_tokens = []
all_tokens_no_stop = []
all_bigrams = []
all_trigrams = []

for q in questions:
    tokens = tokenize(q)
    all_tokens.extend(tokens)
    filtered = [t for t in tokens if t not in STOPWORDS]
    all_tokens_no_stop.extend(filtered)
    all_bigrams.extend(get_ngrams(tokens, 2))
    all_trigrams.extend(get_ngrams(tokens, 3))

vocab_size = len(set(all_tokens))
vocab_no_stop = len(set(all_tokens_no_stop))
print(f"  Total tokens: {len(all_tokens):,}")
print(f"  Vocabulary size (all): {vocab_size:,}")
print(f"  Vocabulary size (no stopwords): {vocab_no_stop:,}")

top_unigrams = Counter(all_tokens_no_stop).most_common(30)
top_bigrams = Counter(all_bigrams).most_common(20)
top_trigrams = Counter(all_trigrams).most_common(20)

print("\n  Top-30 Unigrams (no stopwords):")
for w, c in top_unigrams:
    print(f"    {w:25s} {c:>6,}")

print("\n  Top-20 Bigrams:")
for w, c in top_bigrams:
    print(f"    {w:35s} {c:>6,}")

print("\n  Top-20 Trigrams:")
for w, c in top_trigrams:
    print(f"    {w:45s} {c:>6,}")

# Plot unigrams
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
words_u, counts_u = zip(*top_unigrams[:20])
axes[0].barh(range(len(words_u)), counts_u, color="coral", edgecolor="white")
axes[0].set_yticks(range(len(words_u)))
axes[0].set_yticklabels(words_u)
axes[0].invert_yaxis()
axes[0].set_title("Top-20 Unigrams (no stopwords)")
axes[0].set_xlabel("Count")

words_b, counts_b = zip(*top_bigrams[:15])
axes[1].barh(range(len(words_b)), counts_b, color="mediumpurple", edgecolor="white")
axes[1].set_yticks(range(len(words_b)))
axes[1].set_yticklabels(words_b)
axes[1].invert_yaxis()
axes[1].set_title("Top-15 Bigrams")
axes[1].set_xlabel("Count")

words_t, counts_t = zip(*top_trigrams[:15])
axes[2].barh(range(len(words_t)), counts_t, color="seagreen", edgecolor="white")
axes[2].set_yticks(range(len(words_t)))
axes[2].set_yticklabels(words_t)
axes[2].invert_yaxis()
axes[2].set_title("Top-15 Trigrams")
axes[2].set_xlabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "02_vocabulary_ngrams.png"), dpi=150)
plt.close()
print(f"\n  => Saved: 02_vocabulary_ngrams.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. SỐ LƯỢNG COMPONENT TRONG MỖI CÂU SQL
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("3. SQL COMPONENT COUNT PER QUERY")
print("=" * 70)

SQL_COMPONENTS = [
    "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY",
    "HAVING", "LIMIT", "UNION", "INTERSECT", "EXCEPT",
    "DISTINCT", "LIKE", "BETWEEN", "IN", "EXISTS",
    "NOT", "AND", "OR", "AS",
]


def count_components(sql):
    """Đếm số loại component (clause) xuất hiện trong 1 câu SQL."""
    sql_upper = sql.upper()
    found = set()
    for comp in SQL_COMPONENTS:
        # Dùng word boundary để tránh false positive
        pattern = r'\b' + re.escape(comp) + r'\b'
        if re.search(pattern, sql_upper):
            found.add(comp)
    return found


component_counts = []  # mỗi câu có bao nhiêu component
component_details = Counter()  # gom lại xem mỗi component xuất hiện bao nhiêu lần

for sql in sql_queries:
    found = count_components(sql)
    component_counts.append(len(found))
    for c in found:
        component_details[c] += 1

# Thống kê theo số lượng component
comp_dist = Counter(component_counts)
print("  Distribution of #components per query:")
for k in sorted(comp_dist.keys()):
    pct = comp_dist[k] / N * 100
    print(f"    {k:2d} components: {comp_dist[k]:>6,} queries ({pct:5.2f}%)")

print(f"\n  Mean components/query: {np.mean(component_counts):.2f}")
print(f"  Median: {np.median(component_counts):.1f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar chart: distribution
keys_sorted = sorted(comp_dist.keys())
vals_sorted = [comp_dist[k] for k in keys_sorted]
axes[0].bar(keys_sorted, vals_sorted, color="teal", edgecolor="white")
axes[0].set_title("Distribution of #Components per SQL Query")
axes[0].set_xlabel("Number of Components")
axes[0].set_ylabel("Number of Queries")
for i, (k, v) in enumerate(zip(keys_sorted, vals_sorted)):
    axes[0].text(k, v + N * 0.005, str(v), ha='center', fontsize=8)

# Bar chart: component popularity
comp_sorted = component_details.most_common()
comp_names, comp_vals = zip(*comp_sorted)
axes[1].barh(range(len(comp_names)), comp_vals, color="darkorange", edgecolor="white")
axes[1].set_yticks(range(len(comp_names)))
axes[1].set_yticklabels(comp_names)
axes[1].invert_yaxis()
axes[1].set_title("Component Popularity (how many queries use each)")
axes[1].set_xlabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "03_sql_component_counts.png"), dpi=150)
plt.close()
print(f"  => Saved: 03_sql_component_counts.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. TẦN SUẤT XUẤT HIỆN CỦA CÁC MỆNH ĐỀ SQL (SQL Clause Frequencies)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("4. SQL CLAUSE FREQUENCIES")
print("=" * 70)

CLAUSES_TO_CHECK = [
    "JOIN", "GROUP BY", "ORDER BY", "HAVING", "LIMIT",
    "INTERSECT", "EXCEPT", "UNION", "Nested Subquery"
]


def has_nested_subquery(sql):
    """Kiểm tra có subquery lồng nhau không (SELECT ... trong SELECT ...)."""
    upper = sql.upper()
    # Bỏ SELECT đầu tiên, xem có SELECT nào khác không
    first_select = upper.find("SELECT")
    if first_select == -1:
        return False
    remaining = upper[first_select + 6:]
    return "SELECT" in remaining


clause_freq = {}
for clause in CLAUSES_TO_CHECK:
    if clause == "Nested Subquery":
        count = sum(1 for sql in sql_queries if has_nested_subquery(sql))
    else:
        pattern = r'\b' + re.escape(clause.upper()) + r'\b'
        count = sum(1 for sql in sql_queries if re.search(pattern, sql.upper()))
    clause_freq[clause] = count

print(f"  {'Clause':<20s} {'Count':>8s} {'Ratio':>8s}")
print(f"  {'-'*20} {'-'*8} {'-'*8}")
for clause, count in sorted(clause_freq.items(), key=lambda x: -x[1]):
    ratio = count / N * 100
    print(f"  {clause:<20s} {count:>8,} {ratio:>7.2f}%")

fig, ax = plt.subplots(figsize=(10, 5))
clauses_sorted = sorted(clause_freq.items(), key=lambda x: -x[1])
c_names, c_vals = zip(*clauses_sorted)
c_pcts = [v / N * 100 for v in c_vals]
bars = ax.bar(c_names, c_pcts, color="royalblue", edgecolor="white")
ax.set_title("SQL Clause Frequencies (%)")
ax.set_ylabel("% of queries containing clause")
ax.set_xlabel("SQL Clause")
plt.xticks(rotation=30, ha="right")
for bar, pct in zip(bars, c_pcts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{pct:.1f}%", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "04_sql_clause_frequencies.png"), dpi=150)
plt.close()
print(f"\n  => Saved: 04_sql_clause_frequencies.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ĐỘ SÂU SUBQUERIES VÀ SỐ LƯỢNG BẢNG ĐƯỢC JOIN
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("5. SUBQUERY DEPTH & JOIN TABLE COUNT")
print("=" * 70)


def subquery_depth(sql):
    """Tính độ sâu subquery lồng nhau (dựa trên số lần SELECT xuất hiện)."""
    upper = sql.upper()
    depth = upper.count("SELECT") - 1  # trừ SELECT ngoài cùng
    return max(depth, 0)


def count_join_tables(sql):
    """Đếm số bảng tham gia JOIN.
    Cách tính: 1 (bảng chính trong FROM) + số lần xuất hiện JOIN.
    """
    upper = sql.upper()
    join_count = len(re.findall(r'\bJOIN\b', upper))
    if join_count == 0:
        return 0  # không có JOIN
    return join_count + 1  # bảng FROM + mỗi JOIN thêm 1 bảng


depths = [subquery_depth(sql) for sql in sql_queries]
join_tables = [count_join_tables(sql) for sql in sql_queries]

# Subquery depth stats
depth_dist = Counter(depths)
print("  Subquery Depth Distribution:")
for d in sorted(depth_dist.keys()):
    pct = depth_dist[d] / N * 100
    print(f"    Depth {d}: {depth_dist[d]:>6,} queries ({pct:5.2f}%)")
print(f"  Mean subquery depth: {np.mean(depths):.3f}")
print(f"  Max subquery depth:  {max(depths)}")

# JOIN table stats
join_dist = Counter(join_tables)
print(f"\n  JOIN Table Count Distribution:")
for j in sorted(join_dist.keys()):
    pct = join_dist[j] / N * 100
    label = "no JOIN" if j == 0 else f"{j} tables"
    print(f"    {label:>12s}: {join_dist[j]:>6,} queries ({pct:5.2f}%)")

queries_with_join = [jt for jt in join_tables if jt > 0]
if queries_with_join:
    print(f"\n  Among queries WITH JOIN:")
    print(f"    Mean tables joined: {np.mean(queries_with_join):.2f}")
    print(f"    Median: {np.median(queries_with_join):.1f}")
    print(f"    Max:    {max(queries_with_join)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subquery depth
d_keys = sorted(depth_dist.keys())
d_vals = [depth_dist[k] for k in d_keys]
axes[0].bar(d_keys, d_vals, color="crimson", edgecolor="white")
axes[0].set_title("Subquery Nesting Depth Distribution")
axes[0].set_xlabel("Nesting Depth (0 = no subquery)")
axes[0].set_ylabel("Number of Queries")
for k, v in zip(d_keys, d_vals):
    axes[0].text(k, v + N * 0.003, str(v), ha='center', fontsize=8)

# JOIN table count
j_keys = sorted(join_dist.keys())
j_vals = [join_dist[k] for k in j_keys]
j_labels = ["no JOIN" if k == 0 else str(k) for k in j_keys]
axes[1].bar(j_labels, j_vals, color="forestgreen", edgecolor="white")
axes[1].set_title("Number of Tables in JOIN")
axes[1].set_xlabel("# Tables Joined (0 = no JOIN)")
axes[1].set_ylabel("Number of Queries")
for i, (k, v) in enumerate(zip(j_labels, j_vals)):
    axes[1].text(i, v + N * 0.003, str(v), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "05_subquery_depth_join_tables.png"), dpi=150)
plt.close()
print(f"\n  => Saved: 05_subquery_depth_join_tables.png\n")

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Total samples:                    {N:,}")
print(f"  Avg question length (words):      {np.mean(q_word_lens):.1f}")
print(f"  Vocabulary size (no stopwords):   {vocab_no_stop:,}")
print(f"  Avg SQL components/query:         {np.mean(component_counts):.1f}")
print(f"  Queries with JOIN:                {sum(1 for j in join_tables if j > 0):,} "
      f"({sum(1 for j in join_tables if j > 0)/N*100:.1f}%)")
print(f"  Queries with Nested Subquery:     {clause_freq['Nested Subquery']:,} "
      f"({clause_freq['Nested Subquery']/N*100:.1f}%)")
print(f"  Mean subquery depth:              {np.mean(depths):.3f}")
print(f"  Mean tables joined (when JOIN):   "
      f"{np.mean(queries_with_join):.2f}" if queries_with_join else "N/A")
print(f"\n  All charts saved to: {OUTPUT_DIR}")
print("  DONE!")

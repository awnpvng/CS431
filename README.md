# SEMANTIC TEXT-TO-SQL AND RECOMMENDATION FRAMEWORK

## 1. General Information
* **Project Name:** Semantic Text-to-SQL and Recommendation Framework 
* **Course:** Deep Learning Techniques and Applications (CS431.Q22) 
* **Team Members:**
* Lê Ngọc Thành - 23521443 
* Nguyễn Xuân An - 23520023 
* Trương Hoàng Thành An - 23520032 

## 2. Introduction & Objectives
Traditional database query systems are often rigid and return empty results when an exact item is out of stock. This project proposes a **Generative BI** solution using a parallel retrieval architecture:
* **Text-to-SQL:** Retrieves precise data from the database.
* **Vector Search:** Uses semantic search to suggest similar products when exact matches are unavailable.
* **LLM Synthesis:** Combines data into natural, helpful responses for the user.

## 3. Pipeline Architecture
The system functions as an **Insight Engine** with four main stages:
1. **Insight Reader:** Classifies user intent into `ANALYTIC` (data calculation) or `SEARCH` (product recommendation) .
2. **Module 1 (Text-to-SQL):** Translates questions into SQL queries to fetch a DataFrame .
3. **Module Vector Search:** Performs **Hybrid RAG** (Dense + BM25) using FAISS to find the Top-K similar items .
4. **Module 2 (Response Synthesizer):** Uses an LLM to generate a final response based on the SQL results and similar items .

## 4. Data & Methodology
* **Spider Dataset:** Used for training and evaluating the Text-to-SQL module independently.
* **DS2 Database:** Used for end-to-end system testing with a custom 45-question testcase.
* **Embedding Model:** `all-MiniLM-L6-v2` for semantic representation.
* **Fine-tuning:** Applied QLoRA via the **Unsloth** library to optimize models like Qwen and Llama .
* 
## 5. Resource Optimization
To make large AI models run on consumer hardware, the system uses several optimization techniques:
* **4-Bit Quantization:** Reduces model memory usage significantly while maintaining high performance.
* **Unsloth Integration:** Accelerates training speed and reduces VRAM requirements for local fine-tuning .
* **LoRA Dropout:** Set to 0 to maximize training efficiency during the adaptation process.

## 6. Demo Application (Streamlit)
The system is deployed as a web-based application using **Streamlit** to provide a user-friendly interface:
* **Interactive Chat:** Users can ask questions in natural language.
* **Transparency:** The app displays the generated SQL query and the raw data results (DataFrame) alongside the final AI response .
* **Visual Recommendations:** For `SEARCH` queries, it provides a list of recommended products with reasoning .

## 7. Evaluation Results
### 7.1. Spider Dataset Results (Text-to-SQL Module)

| Model | avg_ex | avg_esm | avg_f1 |
| --- | --- | --- | --- |
| Llama-3.2-3B-Instruct | 0.587 | 0.346 | 0.913 |
| Qwen2.5-Coder-3B-Instruct | 0.578 | 0.337 | 0.914 |
| SQLcoder-7B | 0.619 | 0.411 | 0.913 |
| **Qwen2.5-Coder-7B-Instruct** | **0.707** | **0.448** | **0.947** |
| <br>(Based on 2,147 samples) 
 |  |  |  |

### 7.2. DS2 Custom Testcase Results (45 Questions)

| Model | avg_ex | avg_esm | avg_f1 |
| --- | --- | --- | --- |
| Qwen2.5-Coder-3B-Instruct | 0.422 | 0.13 | 0.873 |
| 
 |  |  |  |

### 7.3. Metric Definitions
* **avg_ex (Execution Accuracy):** The percentage of predicted SQL queries that produce the correct result set.
* **avg_esm (Exact Set Match):** The percentage of queries where the SQL structure perfectly matches the target.
* **avg_f1:** The harmonic mean of precision and recall for SQL components.

## 8. Conclusion & Future Work
### 8.1. Conclusion
* Successfully built a flexible Dual-Track architecture for precise queries and semantic suggestions.
* The fine-tuned Qwen2.5-Coder-7B model showed superior performance in SQL generation .
* The Response Synthesizer effectively understands context to provide clear reasoning.

### 8.2. Future Work
* Optimize **Schema Linking** to handle large databases with hundreds of tables.
* Upgrade to larger models (14B, 32B) for more complex logical reasoning.
* Integrate **Multi-turn Memory** to support ongoing conversations.
* Deepen personalization based on individual user behavior history.



# **Similarity Threshold Experiment**

This project explores how different cosine‑similarity thresholds affect search quality when using sentence embeddings. It demonstrates a core concept in search system design: **loose thresholds return more results (including noise), while strict thresholds return fewer results (but may miss relevant matches).**

The experiment uses the `sentence-transformers` library to embed a small knowledge base of sentences across multiple topics, then evaluates how five test queries perform under three different similarity thresholds.

---

## **📌 Objectives**

- Build intuition for tuning similarity thresholds in semantic search.
- Compare how many results pass thresholds of **0.3**, **0.5**, and **0.7**.
- Observe which relevant results disappear as thresholds become stricter.
- Understand how ambiguous queries behave under different cutoff values.

---

## **📚 Knowledge Base**

The project uses 15 sentences across 3–4 distinct topics:

- Python development  
- Cooking  
- Space science  

These provide a diverse embedding space for meaningful similarity comparisons.

---

## **🔍 Test Queries**

Five queries are defined to represent:

- Clear topic matches  
- Ambiguous or multi‑topic queries  
- Queries with varying semantic closeness  

Each query is compared against all 15 knowledge‑base sentences.

---

## **⚙️ How It Works**

1. Encode all sentences using `all-MiniLM-L6-v2`.
2. Encode each query.
3. Compute cosine similarity between the query and all sentences.
4. Apply thresholds:
   - **0.3** → loose, many results  
   - **0.5** → moderate, fewer results  
   - **0.7** → strict, only highly relevant results  
5. Print:
   - Number of results passing each threshold  
   - Each passing result with its similarity score  

This mirrors real‑world search tuning where threshold selection impacts relevance and noise.

---

## **📄 Example Output**

```
Query: "How do I deploy my Python app?"

  Threshold 0.3: 6 results
    [0.72] Deploy FastAPI with uvicorn and Docker
    [0.65] Python web frameworks comparison
    [0.48] Setting up a virtual environment
    [0.41] Git basics for version control
    [0.35] Writing automated tests with pytest
    [0.31] SQL database migration strategies

  Threshold 0.5: 3 results
    [0.72] Deploy FastAPI with uvicorn and Docker
    [0.65] Python web frameworks comparison

  Threshold 0.7: 1 result
    [0.72] Deploy FastAPI with uvicorn and Docker
```

This demonstrates how stricter thresholds reduce result count and may filter out relevant items.

---

## **🚀 Running the Project**

Install dependencies:

```bash
pip install sentence-transformers numpy
```

Run the script:

```bash
python threshold_experiment.py
```

You’ll see threshold‑based results printed for each query.

---

## **🧠 Why This Matters**

Similarity thresholds are a foundational concept in:

- Search engines  
- Recommendation systems  
- Chatbot retrieval pipelines  
- Document ranking  
- Semantic filtering  

This experiment builds intuition you’ll apply in larger projects, especially when tuning retrieval quality.

---

## **📎 Files**

- `threshold_experiment.py` — main script containing:
  - Knowledge base  
  - Query definitions  
  - Embedding generation  
  - Threshold comparison logic  

---

## **📈 Future Improvements**

If you want to extend the project, consider:

- Adding visualization (matplotlib similarity histograms)
- Adding topic labels to results
- Building a Streamlit UI
- Logging “missed relevant results” at strict thresholds
- Experimenting with different embedding models


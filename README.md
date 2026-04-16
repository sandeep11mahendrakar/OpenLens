# **🚀 OpenLens: Building an ML-Powered Research & Answer Engine**

OpenLens is a full-stack **Retrieval-Augmented Generation (RAG)** system designed to replicate the architecture behind industry leaders like Perplexity, Bing Chat, and Google AI Overviews. This project demonstrates proficiency in ML pipelines, data engineering, and production-grade LLM implementation.

## **🏗️ System Architecture**

The pipeline follows a modular "Intent-to-Answer" flow:

1. **Intent Classifier:** Analyzes the user query to determine the required search strategy.  
2. **Scraping Engine:** Fetches real-time data from diverse sources (Wikipedia, Reddit, etc.).  
3. **RAG Pipeline:** Chunks, embeds, and indexes retrieved data for semantic search.  
4. **Synthesis Engine:** Uses an LLM to generate a coherent, grounded response.  
5. **API & UI:** Serves the engine via a FastAPI backend and a modern frontend.

## **📅 4-Month Implementation Roadmap**

### **MONTH 1: Foundation — Intent & Scraping**

*Focus: Determining "what" the user wants and "where" to find it.*

* **Weeks 1–2: The Intent Classifier**  
  * dataset\_builder.py: Generate and curate synthetic training data for classification.  
  * model.py: Train a classification model (e.g., DistilBERT or Scikit-learn) to categorize queries into modes like *Research*, *Advice*, or *Verify*.  
* **Weeks 3–4: Smart Scraping Engine**  
  * wikipedia\_scraper.py: Optimized retrieval for factual, long-form content.  
  * reddit\_scraper.py: Social-context retrieval, ideal for the *Advice* and *Verify* intent modes.

### **MONTH 2: The Core — Semantic Search Pipeline**

*Focus: Transforming raw text into a searchable ML database.*

* **document\_processor.py**:  
  * **Text Chunking:** Splitting long articles into manageable, semantically meaningful pieces.  
  * **Embedding:** Converting text into high-dimensional vectors using models like sentence-transformers.  
  * **Indexing:** Storing vectors in a Vector DB (e.g., FAISS or Pinecone) for sub-millisecond retrieval.

### **MONTH 3: The Brain — Answer Synthesis**

*Focus: Turning retrieved data into human-readable answers.*

* **answer\_engine.py**:  
  * **Context Injection:** Feeding the top-k retrieved chunks into an LLM (GPT-4o, Claude, or Llama 3).  
  * **Prompt Engineering:** Designing system prompts that ensure the response is grounded in the retrieved facts to minimize hallucinations.  
  * **Citations:** Implementing a mechanism to map parts of the response back to original source URLs.

### **MONTH 4: Production — API, UX & Evaluation**

*Focus: Taking the project from a script to a product.*

* **app/main.py**: A FastAPI backend to serve the full pipeline.  
* **evaluator.py**:  
  * **RAGAS Metrics:** Systematic evaluation of *Faithfulness*, *Answer Relevance*, and *Context Precision*.  
  * **Benchmarking:** Measuring latency and accuracy across different query types.

## **🌟 Stretch Goals (Post-MVP)**

| Enhancement | Difficulty | Impact |
| :---- | :---- | :---- |
| **Cross-Encoder Re-ranker** | Medium | **High** — Significantly improves retrieval accuracy. |
| **Streamlit UI** | Easy | **High** — Provides a visual demo for interviews. |
| **News API Integration** | Easy | **Medium** — Adds real-time current events capability. |
| **MLflow Tracking** | Medium | **High** — Shows professional ML experiment management. |
| **GitHub Actions CI/CD** | Medium | **High** — Automated testing and evaluation on push. |
| **Query Reformulation** | Hard | **Medium** — Rewrites user queries for better search results. |
| **A/B Testing Framework** | Hard | **Very High** — Compares different pipeline versions. |

## **💡 Why This Project?**

Building OpenLens isn't just about calling an API; it is about managing the **data lifecycle**. By completing this roadmap, you demonstrate skills in:

* **NLP & Classification** (Intent Model)  
* **Data Engineering** (Scrapers & Chunking)  
* **Vector Databases** (Semantic Search)  
* **LLM Operations** (Synthesis & Evaluation)
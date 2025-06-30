# 🔍 Context-Aware Missing Value Imputation

**A Novel Hybrid Framework Combining LLMs, Semantic Embeddings, and Vector Databases**



## 📌 Overview

This project introduces a **novel approach to missing value imputation** in tabular datasets using:

* **Google’s Text Embedding models** for contextual row encoding
* **Pinecone Vector DB** for fast similarity-based retrieval
* **LLMs (e.g., Gemini, LLaMA3 via Groq)** for intelligent inference of missing values
* **Validation feedback loop** ensuring data consistency and iterative correction

> Traditional imputation fails to capture data semantics. This framework bridges that gap using **semantic vectorization and generative reasoning**, resulting in **smarter, context-aware imputations**.

---

## ⚙️ Architecture

```
[ Incomplete Tabular Data ]
              ↓
[ Google Embeddings for Rows ]
              ↓
[ Pinecone Vector Search (Top-k Similar Rows) ]
              ↓
[ LLM Inference (Predict Missing Fields) ]
              ↓
[ Validation (Correlation, KS Test, Stats) ]
              ↓
[ Complete Dataset with Imputed Values ]
              ↺ (Iterative Feedback Loop if Validation Fails)
```

---

## 🚀 Features

* ✅ **Context-aware Imputation** using prompt-driven LLM inference
* ⚡ **Efficient Retrieval** via Pinecone similarity search
* 📊 **Validation Modules** to ensure statistical integrity of imputed values
* 🔁 **Iterative Refinement** to reprocess low-quality imputations
* 📦 **Modular Design** for easy extension and deployment

---

## 🧪 Technologies Used

| Component       | Technology                                |
| --------------- | ----------------------------------------- |
| Embedding Model | `text-embedding-004` (Google AI)          |
| Vector Storage  | `Pinecone` Vector DB                      |
| LLM Inference   | `Gemini 2.0` (Google) / `LLaMA3` via Groq |
| Frontend        | `Streamlit` for UI (optional)             |
| Backend         | `Python`, `Flask/FastAPI` (optional)      |

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/context-aware-imputation.git
cd context-aware-imputation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

Create a `.env` file with the following:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the Pipeline

```bash
streamlit run app.py
```

---

## 🧠 How It Works

1. **Preprocess Dataset**: Clean and normalize your dataset.
2. **Generate Embeddings**: Use Google embedding API to vectorize rows.
3. **Store in Pinecone**: Embed complete rows and store them in Pinecone.
4. **Retrieve Similar Rows**: For each incomplete row, retrieve `top-k` similar rows.
5. **LLM Inference**: Construct a prompt with retrieved rows and predict missing fields using Gemini or Groq.
6. **Validation**: Run correlation check, KS test, and distribution comparison.
7. **Iterate if Needed**: If validation fails, reprocess row with new context.

---

## 📈 Evaluation Metrics

* **For Known Truth**: MAE, RMSE, R²
* **For Unsupervised Settings**:

  * **Kolmogorov-Smirnov Test**
  * **Correlation Preservation**
  * **Distributional Similarity Index (SSI)**

---

## 📊 Sample Prompt for LLM

```plaintext
You are a helpful AI assistant.

Given similar complete rows:
[{"Age": 30, "Income": 50K, "Job": "Engineer"}, {"Age": 29, "Income": 52K, "Job": "Developer"}]

And this incomplete row:
{"Age": 30, "Income": "[MISSING]", "Job": "Engineer"}

Predict only the missing values as a JSON dictionary.
```

---

<!-- ## 🧪 Experimental Results

| Dataset    | Method      | MAE      | RMSE     | R²       |
| ---------- | ----------- | -------- | -------- | -------- |
| HealthData | Mean Impute | 8.42     | 9.76     | 0.63     |
| HealthData | Our Method  | **4.18** | **5.22** | **0.84** |

--- -->

<!-- ## 🔄 Future Enhancements

* Add support for multi-modal data (e.g., images + tabular)
* Extend to real-time imputation via REST API
* Integrate feedback-based prompt tuning using validation heuristics
* Explore reinforcement learning-based reward modeling for optimal predictions

--- -->

## 🤝 Contributing

We welcome contributors!

1. Fork the repo
2. Create your feature branch
3. Submit a PR with detailed explanation

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 📣 Acknowledgements

* Pinecone for scalable vector search
* Google Generative AI for embedding + inference models
* Groq for efficient LLaMA3 inference
* Streamlit for visualization and feedback loop interface

---



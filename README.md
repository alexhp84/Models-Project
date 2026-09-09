# Segment Studio — Automated Data Segmentation

An interactive K-Means clustering system that discovers hidden groups
("segments") inside a CSV dataset using unsupervised machine learning,
then uses a self-hosted LLM to give each cluster a human-readable name
and description.

The project has two parts:

- **`Silhouette_Score.py` + `LLM.py`** — a console-based clustering
  pipeline (load → preprocess → choose K → cluster → LLM enrichment →
  export).
- **`app.py`** — a Streamlit web app ("Segment Studio") that wraps the
  same pipeline in a guided, 5-stage UI.

## Pipeline

1. **Load real data** — read the input CSV into a `pandas.DataFrame`.
2. **Preprocessing** — impute missing values and scale numeric
   columns; impute and one-hot encode categorical columns.
3. **Cluster structure analysis** — compute WCSS / silhouette score
   across a user-chosen K range and plot the elbow curve.
4. **K-Means clustering** — fit K-Means for the chosen K (manual or
   automatic) and assign a `cluster_id` to every row.
5. **LLM interpretation** — send each cluster's profile (size, average
   numeric features, most common categorical values) to an LLM, which
   returns a short name and one-line description per cluster.
6. **Export** — write the result back to
   `<original_name>_clustered.csv`, with a new `name_cluster` column
   mapping every row to its segment name.

## Requirements

```bash
pip install -r requirements.txt
```

## Configuration

The LLM step calls a self-hosted, OpenAI-compatible chat completions
endpoint. Copy the template and fill in your own values:

```bash
cp .env.example .env
```

`.env.example`:

```
LLM_API_KEY=your-key
LLM_ENDPOINT=http://your-endpoint/api/chat/completions
LLM_MODEL=your-model-name
```

`.env` itself is not committed (see `.gitignore`).

## Usage

### Console pipeline

```bash
python Silhouette_Score.py
```

You'll be prompted for a K range and a manual/automatic clustering
mode. Output is written to `<input>_clustered.csv`, and the fitted
preprocessor + model are serialized to `cluster_model.joblib`.

### Streamlit app

```bash
streamlit run app.py
```

Walks through the same pipeline in the browser:

1. Upload a CSV.
2. Choose a K range and compute WCSS / silhouette scores.
3. Pick K (manually or automatically) and create clusters.
4. Generate cluster names/descriptions via the LLM.
5. Download the labeled CSV.

## Project structure

```
.
├── app.py                  # Streamlit UI (Segment Studio)
├── Silhouette_Score.py     # Core clustering pipeline
├── LLM.py                  # LLM client (self-hosted endpoint)
├── .env                      # Local secrets (gitignored, not in repo)
├── .env.example             # Template for LLM_API_KEY/ENDPOINT/MODEL
├── requirements.txt
├── iris_unlabaled.csv      # Sample input dataset
└── cluster_model.joblib    # Trained preprocessor + KMeans model
```

## Notes

- `cluster_model.joblib` is excluded by `.gitignore` by default but is
  required for submission — add it explicitly with `git add -f
  cluster_model.joblib` (or remove the `*.joblib` rule) before
  pushing.
- Assignment spec: `project02_clusters_dec25.pdf`.

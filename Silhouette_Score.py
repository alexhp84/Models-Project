import json
import re
import logging
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from LLM import ask_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

#Step 1 - Loading and inspecting the data
#===========================
def load_csv(path: str = "iris_unlabaled.csv"):
    """
    Load the CSV and immediately display the first
    five rows so the user can verify the structure.
    """
    filename = input("Enter the CSV filename: ").strip()

    if not filename:
        raise ValueError("A CSV filename is required.")

    if not Path(filename).is_file():
        raise FileNotFoundError(f"{filename} not found.")

    try:
        df_iris = pd.read_csv(filename)
    except pd.errors.ParserError as e:
        logger.error(f"Could not parse {filename}: {e}")
        raise

    print("\nDATA PREVIEW")
    print(df_iris.head())

    return df_iris, filename

def identify_column_types(df_col_types, exclude_columns):
    """
    Returns lists of numeric and categorical columns after removing
    any in *exclude_columns* (e.g. target, index).
    """
    if exclude_columns is None: exclude_columns = []
    df_clean = df_col_types.drop(columns=exclude_columns, errors="ignore")
    numeric_columns = df_clean.select_dtypes(include="number").columns.tolist()
    categorical_columns = df_clean.select_dtypes(exclude="number").columns.tolist()
    logger.debug(f"Identified columns – numeric ({len(numeric_columns)}), categorical ({len(categorical_columns)})")
    return numeric_columns, categorical_columns

#Step 2 - Silhouette Score & Graph
#===========================/
def build_preprocessing_pipeline(numeric_columns, categorical_columns):
    """
    Construct a ColumnTransformer that imputes the median and
    standard‑scales numeric and categorical columns separately.
    """
    numeric_stage = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
    categorical_stage = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [("numeric", numeric_stage, numeric_columns), ("categorical", categorical_stage, categorical_columns)],
        remainder="drop")

def preprocess_data(df_preprocessed, exclude_columns=None):
    """
    Apply the preprocessing pipeline to *df* and return:
        * the fitted transformer,
        * the transformed DataFrame,
        * the list of numeric column names,
        * the list of categorical column names.
    """
    numeric_columns, categorical_columns = identify_column_types(df_preprocessed, exclude_columns)
    preprocessor = build_preprocessing_pipeline(numeric_columns, categorical_columns)
    transformed_features = preprocessor.fit_transform(df_preprocessed.drop(columns=exclude_columns, errors="ignore"))
    df_processed = pd.DataFrame(transformed_features, columns=preprocessor.get_feature_names_out(), index=df_preprocessed.index,)
    logger.info(f"Pre‑processing completed – transformed shape {df_processed.shape}")
    return preprocessor, df_processed, numeric_columns, categorical_columns

#Step 3 - Choosing K and processing
#===========================

def get_k_range(n_samples, features):
    """
    Ask the user to choose manual or automatic K selection.
    Returns (k_min, k_max, best_k).
    * In manual mode: best_k is None; the caller must pick a value between
      k_min and k_max.
    * In automatic mode: best_k is the silhouette‑optimal cluster count
      (k_min and k_max are still returned for reporting purposes).
    """
    max_allowed = n_samples - 1

    while True:
        try:
            k_min = int(input(f"\nEnter the minimum number of clusters "
                              f"(k_min, 2-{max_allowed}): "))
            if 2 <= k_min <= max_allowed:
                break
            logger.error(f"k_min must be between 2 and {max_allowed}.")
        except ValueError:
            logger.error("Please enter an integer for k_min.")

    while True:
        try:
            k_max = int(input(f"Enter the maximum number of clusters "
                              f"(k_max, {k_min}-{max_allowed}): "))
            if k_min <= k_max <= max_allowed:
                break
            logger.error(f"k_max must be between {k_min} and {max_allowed}.")
        except ValueError:
            logger.error("Please enter an integer for k_max.")

    #Mode selection
    while True:
        mode = input(
            "\nChoose mode:\n"
            "1 – Manual k selection\n"
            "2 – Automatic k selection (silhouette)\n"
            "Enter 1 or 2: ").strip()
        if mode in {"1", "2"}:
            break
        logger.error("Invalid mode selection. Expected '1' or '2'.")

    #Manual mode
    if mode == "1":
        logger.info(f"Selected k-range: {k_min} – {k_max}")
        return k_min, k_max, None

    #Automatic mode
    best_k, best_score = None, -1.0
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        score = silhouette_score(features, kmeans.fit_predict(features))
        if score > best_score:
            best_score, best_k = score, k

    logger.info(f"Automatic mode chose k={best_k} (silhouette={best_score:.4f})")
    return k_min, k_max, best_k

def plot_elbow_silhouette(preprocessed_features, k_min,k_max):
    """
    For each k in fit K‑Means, compute the silhouette
    score and plot the resulting elbow/Silhouette curve.
    """
    cluster_values = range(k_min, k_max + 1)
    silhouette_scores = [silhouette_score(preprocessed_features, KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(preprocessed_features),)
        for k in cluster_values]

    plt.figure(figsize=(7, 4))
    plt.plot(cluster_values, silhouette_scores, "bo-")
    plt.xlabel("K")
    plt.ylabel("Silhouette")
    plt.title("K Means Silhouette Graph")
    plt.show()
    logger.info("Silhouette plot displayed")
    return pd.DataFrame(list(zip(cluster_values, silhouette_scores)), columns=["K", "Silhouette Score"])

def assign_cluster_labels(df_iris, cluster_labels):
    """
    Attach the cluster label vector to the original DataFrame
    and return a count table per cluster.
    """
    df_clustered = df_iris.copy()
    df_clustered["cluster_id"] = cluster_labels
    df_cluster_counts = df_clustered.groupby("cluster_id").size().reset_index(name="count")
    logger.info(f"Cluster IDs assigned – {len(df_cluster_counts)} clusters")
    return df_clustered, df_cluster_counts

def profile_numeric_features(df_clustered, numeric_columns):
    """
    For each cluster compute the mean of every numeric feature.
    """
    df_numeric = df_clustered.groupby("cluster_id")[numeric_columns].mean().reset_index()
    logger.debug("Numeric cluster profile computed")
    return df_numeric

def profile_categorical_features(df_clustered, categorical_columns):
    """
    For each cluster compute the mode of every categorical feature.
    """
    # Handle the case where there are *no* categorical columns
    if not categorical_columns:
        # Log the situation and return an empty DataFrame
        logger.info(
            "No categorical columns provided for profiling. "
            "Returning an empty DataFrame with only 'cluster_id'."
        )
        # The empty dataframe still contains the cluster_id column
        return pd.DataFrame(columns=["cluster_id"])

    # Original logic – compute mode per cluster per categorical column
    df_categorical = pd.concat({categorical_column: df_clustered.groupby("cluster_id")[categorical_column].apply
    (lambda x: x.mode().iloc[0])
            for categorical_column in categorical_columns}, axis=1,).reset_index()
    logger.debug("Categorical cluster profile computed")
    return df_categorical

def build_cluster_df_summary(df_cluster_counts, df_numeric, df_categorical):
    """
    Merge count, numeric, and categorical summaries into a single data
    frame keyed by *cluster_id*.
    """
    df_summary = (df_cluster_counts.merge(df_numeric, on="cluster_id", how="left").merge(df_categorical, on="cluster_id", how="left"))
    logger.info("Cluster summary DataFrame built")
    return df_summary

def perform_kmeans_clustering(df_iris, preprocessed_features, numeric_columns, categorical_columns, k):
    """
    Fit K-Means with *k* clusters on *preprocessed_features*,
    profile the resulting clusters, and return the
    cluster‑labelled dataframe, summary dataframe, and the trained
    KMeans model.
    """
    kmeans_model = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(preprocessed_features)
    df_clustered, df_cluster_counts = assign_cluster_labels(df_iris, kmeans_model.labels_)
    df_numeric = profile_numeric_features(df_clustered, numeric_columns)
    df_categorical = profile_categorical_features(df_clustered, categorical_columns)
    df_summary = build_cluster_df_summary(df_cluster_counts, df_numeric, df_categorical)
    logger.info(f"Clustering completed (k={k})")
    return df_clustered, df_summary, kmeans_model

#Step 4 - Connecting to LLM and enhancing clusters
#===========================

def _parse_llm_json(text):
    """
    Strip Markdown code fences (e.g. ```json ... ```) if present and
    parse the remaining text as JSON.
    """
    cleaned = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(cleaned)

def enrich_clusters_with_llm(df_summary, progress_callback=None):
    """
    Send each cluster’s data to the LLM and populate the
    *short_name* and *description* columns.  The function is resilient
    to index‑type mismatches: if the index is a string (e.g. '\\60')
    it is coerced back to an integer before assignment.
    """
    df_summary["short_name"] = ""
    df_summary["description"] = ""

    for row_index, row_data in df_summary.iterrows():
        if progress_callback:
            progress_callback(f"Working on Cluster {row_data['cluster_id']}")
        prompt = f'''
Cluster: {row_data["cluster_id"]}
Observations: {row_data["count"]}
Attributes: {{
    {' , '.join(f"{c}: {row_data[c]}" for c in row_data.index if c not in ("cluster_id", "count"))}
}}
Give name and description in JSON (keys: short_name, description).
'''
        try:
            llm_response = ask_llm(prompt)
            llm_output: dict = _parse_llm_json(llm_response)
            idx = int(row_index) if isinstance(row_index, str) else row_index
            df_summary.at[idx, "short_name"] = llm_output.get("short_name", "Unknown")
            df_summary.at[idx, "description"] = llm_output.get("description", "LLM error")
            logger.info(f"LLM enriched cluster {row_data['cluster_id']}")
            if progress_callback:
                progress_callback(f"Working on Cluster {row_data['cluster_id']} ✓ Complete")
        except Exception as exc:
            idx = int(row_index) if isinstance(row_index, str) else row_index
            df_summary.at[idx, "short_name"] = "Unknown"
            df_summary.at[idx, "description"] = f"LLM error: {exc}"
            logger.error(f"LLM error for cluster {row_data['cluster_id']}: {exc}")
    return df_summary

#Step 5 - Exporting the final DF
#===========================

def export_final_dataframe(df_iris, df_clustered, df_enriched_summary, output_file ):
    """
    Merge the original data with cluster IDs and their LLM‑generated
    names, output to *output_file*, and return the final dataframe.
    """
    df_final = df_iris.copy()
    df_final["cluster_id"] = df_clustered["cluster_id"]
    df_cluster_names = (df_enriched_summary[["cluster_id", "short_name"]].rename(columns={"short_name": "cluster_name"}))
    df_final = df_final.merge(df_cluster_names, on="cluster_id", how="left")
    df_final.to_csv(output_file, index=False)
    logger.info(f"Final CSV written to {output_file}")
    return df_final

if __name__ == "__main__":
    df_iris = load_csv()
    preprocessor, preprocessed_features, numeric_columns, categorical_columns = preprocess_data(df_iris, exclude_columns=["target", "Unnamed: 0"])
    k_min, k_max, best_k = get_k_range(preprocessed_features.shape[0], preprocessed_features)
    plot_elbow_silhouette(preprocessed_features, k_min, k_max)

    if best_k is None:
        while True:
            try:
                chosen_k = int(input(f"\nBased on the silhouette plot, enter your chosen k ({k_min}-{k_max}): "))
                if k_min <= chosen_k <= k_max:
                    break
                logger.error(f"k must be between {k_min} and {k_max}.")
            except ValueError:
                logger.error("Please enter an integer for k.")
    else:
        chosen_k = best_k

    df_clustered, df_summary, kmeans_model = perform_kmeans_clustering(df_iris, preprocessed_features, numeric_columns, categorical_columns, chosen_k)
    df_summary = enrich_clusters_with_llm(df_summary)
    df_final = export_final_dataframe(df_iris, df_clustered, df_summary)
    joblib.dump({"preprocessor": preprocessor, "model": kmeans_model}, "cluster_model.joblib")
    logger.info("Model and preprocessor serialized to cluster_model.joblib")

    print("\n===== FINAL DATA =====")
    print(df_final.head())
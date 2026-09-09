import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path
from Silhouette_Score import (
    preprocess_data,
    plot_elbow_silhouette,
    perform_kmeans_clustering,
    enrich_clusters_with_llm,
    export_final_dataframe,
)


st.set_page_config(
    page_title="Segment Studio",
    page_icon="🧩",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #071a33;
        color: #f4f7fb;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 26, 51, 0);
    }

    [data-testid="stSidebar"] {
        background-color: #06162b;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .step-nav {
        display: flex;
        gap: 10px;
        margin-bottom: 25px;
    }

    .step-label {
        flex: 1;
        padding: 12px 8px;
        border-radius: 10px;
        background: #102a4a;
        color: #9eb3ca;
        text-align: center;
        font-weight: 600;
    }

    .step-label.active {
        background: #1d4f85;
        color: white;
    }

    .step-label.complete {
        background: #153d65;
        color: #dcecff;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 9px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        background: #0b2342;
        border-radius: 12px;
        padding: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session State
# ============================================================

DEFAULTS = {
    "stage": 1,
    "uploaded_file": None,
    "filename": "",

    "df_original": None,
    "df_features": None,
    "preprocessor": None,

    "numeric_columns": None,
    "categorical_columns": None,

    "k_min": 2,
    "k_max": 6,

    "df_silhouette": None,

    "chosen_k": None,
    "chosen_score": None,
    "k_mode": None,

    "df_clustered": None,
    "df_summary": None,
    "df_enriched_summary": None,
    "df_final": None,

    "llm_available": None,
}


for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# Utility Functions
# ============================================================

def reset_after_file():
    for key in (
        "df_features",
        "preprocessor",
        "numeric_columns",
        "categorical_columns",
        "df_silhouette",
        "chosen_k",
        "chosen_score",
        "k_mode",
        "df_clustered",
        "df_summary",
        "df_enriched_summary",
        "df_final",
        "llm_available",
    ):
        st.session_state[key] = DEFAULTS[key]


def navigate_to(stage):
    st.session_state.stage = stage
    st.rerun()


def display_dataframe(df, title):
    st.subheader(title)
    st.dataframe(df, width="stretch")


# ============================================================
# Top Navigation
# ============================================================

def render_navigation():

    stage = st.session_state.stage

    labels = [
        "1  Upload",
        "2  Choose K",
        "3  Complete",
        "4  Graph & Data",
        "5  Download",
    ]

    enabled = [
        True,
        st.session_state.df_silhouette is not None,
        st.session_state.df_clustered is not None,
        st.session_state.df_clustered is not None,
        st.session_state.df_clustered is not None,
    ]

    cols = st.columns(5)

    for i, label in enumerate(labels):

        with cols[i]:

            if st.button(
                label,
                key=f"top_nav_{i + 1}",
                width="stretch",
                disabled=not enabled[i],
            ):
                navigate_to(i + 1)

    completed = [
        True,
        st.session_state.df_silhouette is not None,
        st.session_state.df_clustered is not None,
        st.session_state.df_clustered is not None,
        st.session_state.df_final is not None,
    ]

    html = '<div class="step-nav">'

    for i, label in enumerate(labels, 1):

        classes = ["step-label"]

        if i == stage:
            classes.append("active")

        elif completed[i - 1]:
            classes.append("complete")

        html += (
            f'<div class="{" ".join(classes)}">'
            f"{label}"
            f"</div>"
        )

    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Silhouette Analysis
# ============================================================

def run_silhouette():

    (
        st.session_state.preprocessor,
        st.session_state.df_features,
        st.session_state.numeric_columns,
        st.session_state.categorical_columns,
    ) = preprocess_data(
        st.session_state.df_original,
        ["target", "Unnamed: 0"]
    )
    st.session_state.df_silhouette = plot_elbow_silhouette(
        st.session_state.df_features,
        st.session_state.k_min,
        st.session_state.k_max,
    )


# ============================================================
# K Selection
# ============================================================

def select_k_auto():

    df = st.session_state.df_silhouette

    row = df.loc[
        df["Silhouette Score"].idxmax()
    ]

    st.session_state.chosen_k = int(row["K"])

    st.session_state.chosen_score = float(
        row["Silhouette Score"]
    )

    st.session_state.k_mode = "automatic"


def select_k_manual(k):

    df = st.session_state.df_silhouette

    row = df[
        df["K"] == k
    ].iloc[0]

    st.session_state.chosen_k = int(k)

    st.session_state.chosen_score = float(
        row["Silhouette Score"]
    )

    st.session_state.k_mode = "manual"


# ============================================================
# Clustering Analysis
# ============================================================

def run_analysis():

    (
        st.session_state.df_clustered,
        st.session_state.df_summary,
        kmeans_model,
    ) = perform_kmeans_clustering(
        st.session_state.df_original,
        st.session_state.df_features,
        st.session_state.numeric_columns,
        st.session_state.categorical_columns,
        int(st.session_state.chosen_k),
    )

    progress_placeholder = st.empty()

    def update_cluster_progress(message):
        progress_placeholder.info(message)

    try:

        st.session_state.df_enriched_summary = (
            enrich_clusters_with_llm(
                st.session_state.df_summary.copy(),
                progress_callback=update_cluster_progress,
            )
        )

        names = (
            st.session_state.df_enriched_summary[
                "short_name"
            ].astype(str)
        )

        st.session_state.llm_available = (
            not names.str.startswith("Unknown").all()
        )

    except Exception:

        st.session_state.df_enriched_summary = (
            st.session_state.df_summary.copy()
        )

        st.session_state.df_enriched_summary[
            "short_name"
        ] = "Unknown"

        st.session_state.df_enriched_summary[
            "description"
        ] = "AI naming was unavailable."

        st.session_state.llm_available = False

    finally:

        progress_placeholder.empty()


# ============================================================
# Silhouette Graph
# ============================================================

def silhouette_figure():

    df = st.session_state.df_silhouette

    fig, ax = plt.subplots(
        figsize=(6, 3)
    )

    ax.plot(
        df["K"],
        df["Silhouette Score"],
        marker="o",
    )

    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("K-Means Silhouette Score")

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    return fig


# ============================================================
# Page Header
# ============================================================

st.title("🧩 *Segment Studio*")

render_navigation()


# ============================================================
# STEP 1 - UPLOAD
# ============================================================

if st.session_state.stage == 1:

    st.subheader("Step 1 - Upload & preview")

    uploaded = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    if uploaded is not None:

        file_changed = (
            st.session_state.uploaded_file is None
            or st.session_state.uploaded_file.name
            != uploaded.name
            or st.session_state.uploaded_file.getvalue()
            != uploaded.getvalue()
        )

        if file_changed:

            st.session_state.uploaded_file = uploaded

            st.session_state.filename = uploaded.name

            reset_after_file()

            try:

                st.session_state.df_original = pd.read_csv(
                    io.BytesIO(
                        uploaded.getvalue()
                    )
                )

            except Exception as exc:

                st.session_state.df_original = None

                st.error(
                    f"❌ Error loading CSV: {exc}"
                )

        if st.session_state.df_original is not None:

            st.success(
                f"✅ CSV **{st.session_state.filename}** loaded."
            )

            display_dataframe(
                st.session_state.df_original.head(10),
                "First 10 rows",
            )

            if st.button(
                "Next →",
                key="upload_next",
                width="stretch",
            ):

                navigate_to(2)

    else:

        st.info(
            "Upload a CSV to start the pipeline."
        )


# ============================================================
# STEP 2 - CHOOSE K
# ============================================================

elif st.session_state.stage == 2:

    st.subheader("Step 2 - Choose K")

    if st.session_state.df_original is None:

        navigate_to(1)

    max_allowed = max(
        2,
        st.session_state.df_original.shape[0] - 1,
    )

    col1, col2 = st.columns(2)

    with col1:

        k_min = st.number_input(
            "Minimum K",
            min_value=2,
            max_value=max_allowed,
            value=min(2, max_allowed),
            step=1,
            key="k_min_input",
        )

    with col2:

        default_max = min(
            max(int(k_min) + 4, int(k_min)),
            max_allowed,
        )

        k_max = st.number_input(
            "Maximum K",
            min_value=int(k_min),
            max_value=max_allowed,
            value=default_max,
            step=1,
            key="k_max_input",
        )

    if st.button(
        "ANALYSE",
        key="analyse_button",
        width="stretch",
    ):

        st.session_state.k_min = int(k_min)

        st.session_state.k_max = int(k_max)

        with st.spinner(
            "Computing silhouette scores…"
        ):

            try:

                run_silhouette()

                st.success(
                    "✅ Analysis complete."
                )

            except Exception as exc:

                st.error(
                    f"❌ Analysis failed: {exc}"
                )

    if st.session_state.df_silhouette is not None:

        st.divider()

        st.subheader("Choose K")

        manual_k = st.slider(
            "Manually select K",
            min_value=st.session_state.k_min,
            max_value=st.session_state.k_max,
            value=(
                st.session_state.chosen_k
                if (
                    st.session_state.k_mode == "manual"
                    and st.session_state.chosen_k is not None
                )
                else st.session_state.k_min
            ),
            key="manual_k_slider",
        )

        col1, col2 = st.columns(2)

        with col1:

            if st.button(
                "Use Manual K",
                key="manual_k_button",
                width="stretch",
            ):

                select_k_manual(
                    manual_k
                )

        with col2:

            if st.button(
                "Auto Pick K",
                key="auto_k_button",
                width="stretch",
            ):

                select_k_auto()

        if st.session_state.chosen_k is not None:

            mode_text = (
                "Automatic K"
                if st.session_state.k_mode == "automatic"
                else "Manual K"
            )

            st.success(
                f"{mode_text}: "
                f"**{st.session_state.chosen_k}** "
                f"(Silhouette Score "
                f"**{st.session_state.chosen_score:.4f}**)"
            )

            if st.button(
                "Next →",
                key="choose_k_next",
                width="stretch",
            ):

                navigate_to(3)

    if st.button(
        "← Back",
        key="choose_k_back",
        width="stretch",
    ):

        navigate_to(1)


# ============================================================
# STEP 3 - COMPLETE
# ============================================================

elif st.session_state.stage == 3:

    st.subheader(
        "Step 3 - Analysis Complete"
    )

    if (
        st.session_state.df_silhouette is None
        or st.session_state.chosen_k is None
    ):

        st.warning(
            "Choose K before running the clustering analysis."
        )

        if st.button(
            "← Back",
            key="complete_back_missing",
            width="stretch",
        ):

            navigate_to(2)

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Selected K",
                st.session_state.chosen_k,
            )

        with col2:

            st.metric(
                "Silhouette Score",
                f"{st.session_state.chosen_score:.4f}",
            )

        if st.button(
            "ANALYSE CLUSTERS",
            key="cluster_analysis_button",
            width="stretch",
        ):

            with st.spinner(
                "Creating clusters and generating names…"
            ):

                run_analysis()

            if st.session_state.llm_available:

                st.success(
                    "✅ Clustering analysis complete."
                )

            else:

                st.warning(
                    "AI naming was unavailable. "
                    "The clustering results are still ready to export."
                )

        if st.session_state.df_clustered is not None:

            st.success(
                "Clustering results are ready."
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "← Back",
                    key="complete_back",
                    width="stretch",
                ):

                    navigate_to(2)

            with col2:

                if st.button(
                    "Next →",
                    key="complete_next",
                    width="stretch",
                ):

                    navigate_to(4)


# ============================================================
# STEP 4 - GRAPH & DATA
# ============================================================

elif st.session_state.stage == 4:

    st.subheader(
        "Step 4 - Graph & Data"
    )

    if st.session_state.df_silhouette is not None:

        st.pyplot(
            silhouette_figure(),
            width="content",
        )

    if st.session_state.df_summary is not None:

        display_dataframe(
            st.session_state.df_summary,
            "Cluster Summary",
        )

    if (
        st.session_state.df_enriched_summary
        is not None
    ):

        display_dataframe(
            st.session_state.df_enriched_summary,
            "Cluster Names & Descriptions",
        )

    if st.session_state.df_clustered is not None:

        display_dataframe(
            st.session_state.df_clustered,
            "Clustered Data",
        )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "← Back",
            key="graph_back",
            width="stretch",
        ):

            navigate_to(3)

    with col2:

        if st.button(
            "Next →",
            key="graph_next",
            width="stretch",
        ):

            navigate_to(5)


# ============================================================
# STEP 5 - DOWNLOAD
# ============================================================

elif st.session_state.stage == 5:

    st.subheader(
        "Step 5 - Download"
    )

    if st.session_state.df_clustered is None:

        st.warning(
            "No clustering results are available yet."
        )

    else:

        if (
            st.session_state.df_enriched_summary
            is None
        ):

            st.session_state.df_enriched_summary = (
                st.session_state.df_summary.copy()
            )

            st.session_state.df_enriched_summary[
                "short_name"
            ] = "Unknown"

            st.session_state.df_enriched_summary[
                "description"
            ] = "AI naming was unavailable."

        if st.session_state.df_final is None:
            output_filename = (
                f"{Path(st.session_state.filename).stem}_clustered.csv"
            )

            st.session_state.df_final = export_final_dataframe(
                st.session_state.df_original,
                st.session_state.df_clustered,
                st.session_state.df_enriched_summary,
                output_filename,
            )

        display_dataframe(
            st.session_state.df_final,
            "Final Dataset",
        )

        csv_data = (
            st.session_state.df_final
            .to_csv(index=False)
            .encode("utf-8")
        )

        output_filename = (
            f"{Path(st.session_state.filename).stem}_clustered.csv"
        )

        st.download_button(
            f"Download {output_filename}",
            data=csv_data,
            file_name=output_filename,
            mime="text/csv",
            width="stretch",
        )
    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "← Back",
            key="download_back",
            width="stretch",
        ):

            navigate_to(4)

    with col2:

        if st.button(
            "Start Again",
            key="start_again",
            width="stretch",
        ):

            for key, value in DEFAULTS.items():

                st.session_state[key] = value

            navigate_to(1)
import json
import os

import pandas as pd
import streamlit as st

from main import (
    CFG,
    LABELS,
    aggregate_aspects,
    analyze_review,
    to_sentiment,
)


# ---------- Page configuration ----------

st.set_page_config(
    page_title="Customer Review Analyzer",
    page_icon="🛒",
    layout="centered",
)

st.title("🛒 Customer Review Analyzer")

st.caption(
    "LLM-based aspect-level sentiment analysis "
    "of Amazon food reviews"
)


paths = CFG["paths"]


# ---------- Helper functions ----------

ICON = {
    "positive": "🟢",
    "negative": "🔴",
    "neutral": "⚪",
}


@st.cache_data
def load_results():
    """
    Load previously generated batch-analysis results.

    Returns None if batch results do not exist yet.
    """

    if not os.path.exists(paths["results_csv"]):
        return None

    df = pd.read_csv(
        paths["results_csv"]
    ).fillna("")

    if "aspects" in df.columns:
        df["aspects"] = df["aspects"].apply(
            lambda x: json.loads(x)
            if isinstance(x, str) and x.strip()
            else []
        )

    # Backward compatibility for older result files.
    if "analysis_status" not in df.columns:
        df["analysis_status"] = "success"

    df["true_sentiment"] = (
        df["original_rating"]
        .apply(to_sentiment)
    )

    df["pred_label"] = (
        df["sentiment"]
        .replace({"mixed": "neutral"})
    )

    df["pred_score"] = (
        df["score"]
        .apply(to_sentiment)
    )

    return df


df = load_results()


# ---------- Tabs ----------

tab_live, tab_results, tab_report = st.tabs(
    [
        "Analyze a review",
        "Results",
        "Business report",
    ]
)


# ============================================================
# LIVE REVIEW ANALYSIS
# ============================================================

with tab_live:

    st.subheader("Analyze a customer review")

    st.write(
        "Enter a review below to perform real-time "
        "LLM-based NLP analysis."
    )

    # Show dataset examples only when results exist.
    if df is not None:

        successful_reviews = df[
            df["analysis_status"] == "success"
        ]

        if not successful_reviews.empty:

            sample = st.selectbox(
                "Or choose a sample review",
                [""] + successful_reviews[
                    "review"
                ].tolist(),
                format_func=lambda text: (
                    text[:80] + "..."
                    if text
                    else "Select a review"
                ),
            )

        else:
            sample = ""

    else:
        sample = ""

    review_text = st.text_area(
        "Review text",
        value=sample,
        height=150,
        placeholder=(
            "Example: The taste is excellent, "
            "but the packaging was damaged."
        ),
    )

    analyze_button = st.button(
        "Analyze Review",
        type="primary",
    )

    if analyze_button:

        if not review_text.strip():

            st.warning(
                "Please enter a customer review."
            )

        elif not os.getenv("LLM_API_KEY"):

            st.error(
                "LLM_API_KEY is not configured. "
                "Add it to your .env file."
            )

        else:

            with st.spinner(
                "Analyzing review with the LLM..."
            ):

                result = analyze_review(
                    review_text
                )

            if result["analysis_status"] == "failed":

                st.error(
                    "The review could not be analyzed. "
                    "Please try again."
                )

            else:

                # ---------- Overall analysis ----------

                st.subheader("Overall Analysis")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Overall Sentiment",
                    result[
                        "sentiment"
                    ].title(),
                )

                c2.metric(
                    "Estimated Rating",
                    f"{result['score']} / 5",
                )

                st.markdown(
                    f"**Summary:** "
                    f"{result['summary']}"
                )

                # ---------- Aspect analysis ----------

                st.subheader(
                    "Aspect-Level Sentiment"
                )

                aspects = result["aspects"]

                if not aspects:

                    st.info(
                        "No specific product aspects "
                        "were identified."
                    )

                else:

                    for aspect in aspects:

                        sentiment = aspect[
                            "sentiment"
                        ]

                        icon = ICON.get(
                            sentiment,
                            "⚪",
                        )

                        st.markdown(
                            f"### {icon} "
                            f"{aspect['aspect'].title()}"
                        )

                        st.write(
                            f"**Sentiment:** "
                            f"{sentiment.title()}"
                        )

                        st.write(
                            f"**Evidence:** "
                            f"“{aspect['evidence']}”"
                        )

                # ---------- Business action ----------

                st.subheader(
                    "Suggested Business Action"
                )

                if (
                    result["action"]
                    and result["action"].lower()
                    != "none"
                ):

                    st.info(
                        result["action"]
                    )

                else:

                    st.write(
                        "No specific action required."
                    )


# ============================================================
# BATCH RESULTS
# ============================================================

with tab_results:

    st.subheader(
        "Batch Analysis Results"
    )

    if df is None:

        st.info(
            "Batch results are not available yet."
        )

        st.write(
            "Run the following commands to generate "
            "the evaluation dataset:"
        )

        st.code(
            "python prepare_data.py\n"
            "python main.py",
            language="bash",
        )

    else:

        successful_df = df[
            df["analysis_status"] == "success"
        ].copy()

        failed_count = (
            len(df) - len(successful_df)
        )

        if successful_df.empty:

            st.warning(
                "No successful LLM analyses are "
                "available."
            )

        else:

            # ---------- Evaluation metrics ----------

            sentiment_accuracy = (
                successful_df[
                    "true_sentiment"
                ]
                == successful_df[
                    "pred_label"
                ]
            ).mean()

            score_accuracy = (
                successful_df[
                    "true_sentiment"
                ]
                == successful_df[
                    "pred_score"
                ]
            ).mean()

            mae = (
                successful_df["score"]
                - successful_df[
                    "original_rating"
                ]
            ).abs().mean()

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Reviews analyzed",
                len(successful_df),
            )

            c2.metric(
                "Rating-to-sentiment accuracy",
                f"{score_accuracy:.1%}",
            )

            c3.metric(
                "Mean rating error",
                f"{mae:.2f} stars",
            )

            st.caption(
                f"Direct LLM sentiment-label "
                f"accuracy: {sentiment_accuracy:.1%}. "
                f"Rating-to-sentiment accuracy maps "
                f"the LLM's 1–5 score into negative, "
                f"neutral, or positive."
            )

            if failed_count:

                st.warning(
                    f"{failed_count} review(s) "
                    f"could not be analyzed and were "
                    f"excluded from evaluation."
                )

            # ---------- Sentiment distribution ----------

            st.subheader(
                "Overall Sentiment Distribution"
            )

            order = [
                "positive",
                "mixed",
                "neutral",
                "negative",
            ]

            sentiment_counts = (
                successful_df[
                    "sentiment"
                ]
                .value_counts()
                .reindex(
                    order,
                    fill_value=0,
                )
            )

            st.bar_chart(
                sentiment_counts
            )

            # ---------- Aspect analysis ----------

            st.subheader(
                "What Customers Talk About"
            )

            aspect_df = (
                aggregate_aspects(
                    successful_df
                )
            )

            if aspect_df.empty:

                st.info(
                    "No aspects were extracted."
                )

            else:

                chart_data = (
                    aspect_df
                    .head(8)
                    .set_index("aspect")
                )

                st.bar_chart(
                    chart_data[
                        [
                            "positive",
                            "negative",
                            "neutral",
                        ]
                    ]
                )

                st.caption(
                    "Each count represents an "
                    "aspect mention extracted from "
                    "a customer review."
                )

            # ---------- Confusion matrix ----------

            with st.expander(
                "Confusion Matrix "
                "(LLM Score vs Star Rating)"
            ):

                cm = pd.crosstab(
                    successful_df[
                        "true_sentiment"
                    ],
                    successful_df[
                        "pred_score"
                    ],
                ).reindex(
                    index=LABELS,
                    columns=LABELS,
                    fill_value=0,
                )

                cm.index = [
                    f"true: {label}"
                    for label in cm.index
                ]

                cm.columns = [
                    f"predicted: {label}"
                    for label in cm.columns
                ]

                st.dataframe(
                    cm,
                    use_container_width=True,
                )


# ============================================================
# BUSINESS REPORT
# ============================================================

with tab_report:

    st.subheader(
        "Business Intelligence Report"
    )

    if not os.path.exists(
        paths["report_md"]
    ):

        st.info(
            "No business report is available yet. "
            "Run python main.py first."
        )

    else:

        with open(
            paths["report_md"],
            encoding="utf-8",
        ) as f:

            report = f.read()

        st.markdown(report)

        st.download_button(
            "Download Report",
            report,
            file_name="customer_review_report.md",
            mime="text/markdown",
        )
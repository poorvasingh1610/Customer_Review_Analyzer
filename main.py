import json
import os
import time
from collections import defaultdict

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import classification_report, confusion_matrix


load_dotenv()


# ---------- Configuration ----------

def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_yaml("config.yaml")
PROMPTS = load_yaml("prompts.yaml")

client = OpenAI(
    api_key=os.getenv("LLM_API_KEY") or "missing",
    base_url=CFG["llm"]["base_url"],
)

ALIASES = CFG.get("aspect_aliases", {})

LABELS = ["negative", "neutral", "positive"]

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "mixed",
}

VALID_ASPECT_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
}


# ---------- Aspect normalization ----------

def normalize_aspect(name):
    """
    Convert aspect names to lowercase and merge simple synonyms.
    Example: flavor -> taste, cost -> price.
    """
    name = str(name).strip().lower()
    return ALIASES.get(name, name)


# ---------- LLM API ----------

def call_llm(user_prompt, system_key, json_mode=False):
    """
    Send a prompt to the configured LLM API.

    Retries failed API calls using exponential backoff.
    """

    llm = CFG["llm"]

    extra = {}

    if json_mode:
        extra["response_format"] = {
            "type": "json_object"
        }

    for attempt in range(1, llm["max_retries"] + 1):

        try:
            response = client.chat.completions.create(
                model=llm["model"],
                temperature=llm["temperature"],
                max_tokens=llm["max_tokens"],
                reasoning_effort=llm["reasoning_effort"],
                messages=[
                    {
                        "role": "system",
                        "content": PROMPTS[system_key],
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                **extra,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("LLM returned an empty response.")

            return content

        except Exception as e:

            if attempt == llm["max_retries"]:
                break

            wait = llm["delay_seconds"] * attempt

            print(
                f"  API error: {e}\n"
                f"  Retry {attempt}/{llm['max_retries']} "
                f"in {wait}s"
            )

            time.sleep(wait)

    raise RuntimeError(
        "LLM call failed after all retry attempts."
    )


# ---------- LLM output validation ----------

def validate(result):
    """
    Validate and clean the structured JSON returned by the LLM.
    """

    required_keys = {
        "sentiment",
        "score",
        "aspects",
        "summary",
        "action",
    }

    if not isinstance(result, dict):
        raise ValueError("LLM output is not a JSON object.")

    if not required_keys.issubset(result):
        missing = required_keys - set(result)
        raise ValueError(
            f"Missing required keys: {missing}"
        )

    # Validate overall sentiment
    sentiment = str(result["sentiment"]).strip().lower()

    if sentiment not in VALID_SENTIMENTS:
        raise ValueError(
            f"Invalid sentiment: {sentiment}"
        )

    # Validate score
    try:
        score = int(result["score"])
    except (TypeError, ValueError):
        raise ValueError("Score must be an integer from 1 to 5.")

    if not 1 <= score <= 5:
        raise ValueError(
            f"Score must be between 1 and 5, got {score}."
        )

    # Validate aspects
    if not isinstance(result["aspects"], list):
        raise ValueError("Aspects must be a list.")

    cleaned_aspects = []

    for aspect in result["aspects"]:

        if not isinstance(aspect, dict):
            continue

        if not {
            "aspect",
            "sentiment",
            "evidence",
        }.issubset(aspect):
            continue

        aspect_name = normalize_aspect(aspect["aspect"])
        aspect_sentiment = str(
            aspect["sentiment"]
        ).strip().lower()
        evidence = str(
            aspect["evidence"]
        ).strip()

        if not aspect_name:
            continue

        if aspect_sentiment not in VALID_ASPECT_SENTIMENTS:
            continue

        if not evidence:
            continue

        cleaned_aspects.append(
            {
                "aspect": aspect_name,
                "sentiment": aspect_sentiment,
                "evidence": evidence,
            }
        )

    # Clean text fields
    summary = str(result["summary"]).strip()
    action = str(result["action"]).strip()

    return {
        "sentiment": sentiment,
        "score": score,
        "aspects": cleaned_aspects,
        "summary": summary,
        "action": action,
        "analysis_status": "success",
    }


# ---------- Fallback ----------

def fallback(error_message="Unknown error"):
    """
    Return a clearly marked failed analysis.

    Failed analyses are not treated as genuine neutral reviews
    during evaluation.
    """

    return {
        "sentiment": "",
        "score": None,
        "aspects": [],
        "summary": "",
        "action": "",
        "analysis_status": "failed",
        "error": str(error_message),
    }


# ---------- Single review analysis ----------

def analyze_review(review):
    """
    Analyze one review using the LLM.
    """

    prompt = PROMPTS["analyze_review"].replace(
        "{review}",
        str(review),
    )

    try:

        raw_response = call_llm(
            prompt,
            "system_analyze",
            json_mode=True,
        )

        parsed_response = json.loads(raw_response)

        return validate(parsed_response)

    except Exception as e:

        print(
            f"  Could not process review: {e}"
        )

        return fallback(e)


# ---------- Aspect aggregation ----------

def aggregate_aspects(df):
    """
    Count positive, negative and neutral mentions
    for each extracted aspect.
    """

    stats = defaultdict(
        lambda: {
            "positive": 0,
            "negative": 0,
            "neutral": 0,
        }
    )

    successful_df = df[
        df["analysis_status"] == "success"
    ]

    for aspects in successful_df["aspects"]:

        for aspect in aspects:

            name = normalize_aspect(
                aspect["aspect"]
            )

            sentiment = aspect["sentiment"]

            stats[name][sentiment] += 1

    rows = []

    for aspect, values in stats.items():

        rows.append(
            {
                "aspect": aspect,
                **values,
                "total": sum(values.values()),
            }
        )

    columns = [
        "aspect",
        "positive",
        "negative",
        "neutral",
        "total",
    ]

    if not rows:
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("total", ascending=False)
        .reset_index(drop=True)
    )


# ---------- Rating to sentiment ----------

def to_sentiment(stars):
    """
    Convert Amazon's 1-5 star rating into
    a three-class sentiment label.
    """

    if stars <= 2:
        return "negative"

    if stars == 3:
        return "neutral"

    return "positive"


# ---------- Evaluation ----------

def evaluate(df):
    """
    Compare LLM predictions against the original
    Amazon star-rating based sentiment labels.
    """

    # Evaluate only successful LLM analyses.
    ev = df[
        df["analysis_status"] == "success"
    ].copy()

    if ev.empty:
        raise ValueError(
            "No successful analyses available for evaluation."
        )

    ev = ev[
        [
            "review",
            "original_rating",
            "sentiment",
            "score",
        ]
    ].copy()

    # Ground truth derived from Amazon star rating.
    ev["true_sentiment"] = (
        ev["original_rating"]
        .apply(to_sentiment)
    )

    # Method A:
    # Direct LLM sentiment classification.
    ev["pred_sentiment"] = (
        ev["sentiment"]
        .replace({"mixed": "neutral"})
    )

    # Method B:
    # Convert LLM 1-5 score to sentiment.
    ev["pred_from_score"] = (
        ev["score"]
        .apply(to_sentiment)
    )

    ev["correct"] = (
        ev["true_sentiment"]
        == ev["pred_sentiment"]
    )

    ev["correct_from_score"] = (
        ev["true_sentiment"]
        == ev["pred_from_score"]
    )

    ev.to_csv(
        CFG["paths"]["evaluation_csv"],
        index=False,
    )

    # Rating prediction error.
    mae = (
        ev["score"]
        - ev["original_rating"]
    ).abs().mean()

    method_a_accuracy = (
        ev["correct"].mean()
    )

    method_b_accuracy = (
        ev["correct_from_score"].mean()
    )

    text = (
        f"Successful analyses: {len(ev)}\n\n"
        f"Method A - LLM sentiment label "
        f"(mixed counted as neutral): "
        f"{method_a_accuracy:.1%}\n"
        f"Method B - LLM 1-5 score mapped "
        f"to sentiment: "
        f"{method_b_accuracy:.1%}\n"
        f"Mean absolute error "
        f"(LLM score vs original rating): "
        f"{mae:.2f} stars\n\n"
    )

    # Detailed classification reports.
    for name, column in [
        ("Method A", "pred_sentiment"),
        ("Method B", "pred_from_score"),
    ]:

        report = classification_report(
            ev["true_sentiment"],
            ev[column],
            labels=LABELS,
            zero_division=0,
        )

        matrix = confusion_matrix(
            ev["true_sentiment"],
            ev[column],
            labels=LABELS,
        )

        cm = pd.DataFrame(
            matrix,
            index=[
                f"true_{label}"
                for label in LABELS
            ],
            columns=[
                f"pred_{label}"
                for label in LABELS
            ],
        )

        text += (
            f"--- {name} ---\n"
            f"{report}\n"
            f"Confusion matrix:\n"
            f"{cm.to_string()}\n\n"
        )

    with open(
        CFG["paths"]["metrics_txt"],
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text)

    print("\n" + text)

    return {
        "successful_analyses": len(ev),
        "sentiment_accuracy": method_a_accuracy,
        "score_sentiment_accuracy": method_b_accuracy,
        "mae": mae,
    }


# ---------- Business report ----------

def generate_report(df, aspect_df):
    """
    Use the LLM to convert aggregated NLP results
    into a concise business report.
    """

    successful_df = df[
        df["analysis_status"] == "success"
    ]

    dist = json.dumps(
        successful_df[
            "sentiment"
        ].value_counts().to_dict()
    )

    aspect_text = (
        aspect_df
        .head(12)
        .to_csv(index=False)
    )

    actions = [
        action
        for action in successful_df.loc[
            successful_df["sentiment"] != "positive",
            "action",
        ]
        if action
        and action.lower() != "none"
    ][:20]

    prompt = (
        PROMPTS["report"]
        .replace(
            "{n}",
            str(len(successful_df)),
        )
        .replace(
            "{sentiment_dist}",
            dist,
        )
        .replace(
            "{aspect_stats}",
            aspect_text,
        )
        .replace(
            "{actions}",
            "\n".join(
                f"- {action}"
                for action in actions
            ),
        )
    )

    return call_llm(
        prompt,
        "system_report",
    )


# ---------- Main pipeline ----------

def main():

    if not os.getenv("LLM_API_KEY"):
        raise SystemExit(
            "LLM_API_KEY not found. "
            "Add your API key to the .env file."
        )

    paths = CFG["paths"]

    os.makedirs(
        "output",
        exist_ok=True,
    )

    reviews = pd.read_csv(
        CFG["data"]["sample_csv"]
    ).fillna("")

    results = []

    for i, review_text in enumerate(
        reviews["review"],
        1,
    ):

        print(
            f"Analyzing review "
            f"{i}/{len(reviews)}"
        )

        result = analyze_review(
            review_text
        )

        results.append(result)

        # Delay between API requests.
        time.sleep(
            CFG["llm"]["delay_seconds"]
        )

    results_df = pd.DataFrame(
        results
    )

    df = pd.concat(
        [
            reviews.reset_index(drop=True),
            results_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Save JSON aspect lists safely inside CSV.
    out = df.copy()

    out["aspects"] = out[
        "aspects"
    ].apply(json.dumps)

    out.to_csv(
        paths["results_csv"],
        index=False,
    )

    # Evaluation.
    metrics = evaluate(df)

    # Aspect aggregation.
    aspect_df = aggregate_aspects(df)

    aspect_df.to_csv(
        paths["aspects_csv"],
        index=False,
    )

    print(
        "Top aspects:\n"
        + aspect_df.head(10).to_string(
            index=False
        )
    )

    # Generate business report.
    print(
        "\nGenerating business report..."
    )

    report = generate_report(
        df,
        aspect_df,
    )

    with open(
        paths["report_md"],
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print(
        "\nPipeline completed successfully."
    )

    print(
        f"Successful analyses: "
        f"{metrics['successful_analyses']}"
    )

    print(
        f"Sentiment accuracy: "
        f"{metrics['sentiment_accuracy']:.1%}"
    )

    print(
        f"Rating-to-sentiment accuracy: "
        f"{metrics['score_sentiment_accuracy']:.1%}"
    )

    print(
        f"Mean absolute error: "
        f"{metrics['mae']:.2f} stars"
    )

    print(
        f"\nOutputs saved in: "
        f"{os.path.dirname(paths['report_md'])}/"
    )


if __name__ == "__main__":
    main()
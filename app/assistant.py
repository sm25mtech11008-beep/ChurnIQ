"""Phase E: retention analyst assistant — Claude with two data tools.

    export ANTHROPIC_API_KEY=...   (or `ant auth login`)
    python app/assistant.py "Which contract type has the riskiest customers?"
    python app/assistant.py            # interactive REPL

Architecture: the Claude API tool runner drives the agentic loop; the model
decides when to call the two tools (single-customer lookup, aggregate stats)
and composes the answer from their results. Requires scores.parquet from
`python -m churniq.batch_score`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anthropic
import pandas as pd
from anthropic import beta_tool

from churniq import config

client = anthropic.Anthropic()

_scores: pd.DataFrame | None = None


def scores() -> pd.DataFrame:
    global _scores
    if _scores is None:
        _scores = pd.read_parquet(config.SCORES_PARQUET)
    return _scores


@beta_tool
def lookup_customer(customer_id: str) -> str:
    """Look up one customer's churn score and details by their customer ID.

    Args:
        customer_id: The customer ID, e.g. CUST0000000042.
    """
    row = scores().loc[scores()["customer_id"] == customer_id]
    if row.empty:
        return f"No customer found with id {customer_id}."
    return row.iloc[0].to_json()


@beta_tool
def aggregate_stats(group_by: str = "risk_tier") -> str:
    """Aggregate churn-risk statistics across all 1M scored customers.

    Args:
        group_by: Column to group by — one of "risk_tier" or "contract".
    """
    if group_by not in ("risk_tier", "contract"):
        return "Error: group_by must be 'risk_tier' or 'contract'."
    agg = (
        scores()
        .groupby(group_by, observed=True)
        .agg(
            customers=("churn_probability", "size"),
            avg_churn_probability=("churn_probability", "mean"),
            flagged_for_retention=("target_for_retention", "sum"),
            avg_monthly_charge=("monthlycharges", "mean"),
        )
        .round(4)
    )
    return agg.to_json()


SYSTEM = """You are ChurnIQ's retention analyst assistant. You answer questions
about customer churn risk using the provided tools over the scored customer
base (1M telco customers, calibrated churn probabilities, business threshold
tuned for retention-campaign ROI). Use the tools for any factual claim about
customers or aggregates — don't guess numbers. Keep answers concise and
actionable for a retention team."""


def ask(question: str) -> str:
    runner = client.beta.messages.tool_runner(
        model="claude-opus-5",
        max_tokens=16000,
        system=SYSTEM,
        tools=[lookup_customer, aggregate_stats],
        messages=[{"role": "user", "content": question}],
    )
    final = runner.until_done()
    return "".join(block.text for block in final.content if block.type == "text")


def main():
    if not config.SCORES_PARQUET.exists():
        sys.exit("No scores found — run `python -m churniq.batch_score` first.")
    if len(sys.argv) > 1:
        print(ask(" ".join(sys.argv[1:])))
        return
    print("ChurnIQ assistant — ask about churn risk (Ctrl+C to exit)")
    while True:
        try:
            question = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if question:
            print(ask(question))


if __name__ == "__main__":
    main()

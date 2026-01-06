from datetime import date

import pandas as pd

from portfolio_snapshot import make_client
from analytics_core import summarize_contributions_vs_equity


def main():
    client = make_client()

    # Adjust this if your Schwab history is shorter/longer
    # start_date = date(2015, 1, 1)
    start_date =  "2022-01-01"

    print("=== Historical Contribution vs Equity Report ===\n")
    print(f"Using cash movements starting {start_date} ...\n")

    summary_df = summarize_contributions_vs_equity(client, start_date=start_date)

    if summary_df.empty:
        print("No data found (no deposits/withdrawals or API returned nothing).")
        return

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

    cols = [
        "account_number",
        "account_type",
        "equity",
        "net_contribution",
        "pnl_total",
        "pnl_pct",
    ]
    present_cols = [c for c in cols if c in summary_df.columns]

    print("Per-account view:\n")
    print(summary_df[present_cols].to_string(index=False))

    # Totals across all accounts
    total_equity = summary_df["equity"].sum()
    total_contrib = summary_df["net_contribution"].sum()
    total_pnl = total_equity - total_contrib
    total_pnl_pct = None
    if total_contrib != 0:
        total_pnl_pct = total_pnl / total_contrib * 100.0

    print("\n=== Aggregated across ALL accounts ===")
    print(f"Total net contributions: ${total_contrib:,.2f}")
    print(f"Total current equity:    ${total_equity:,.2f}")
    print(f"TOTAL P&L (all time):    ${total_pnl:,.2f}")
    if total_pnl_pct is not None:
        print(f"TOTAL P&L % vs contrib:  {total_pnl_pct:,.2f} %")

    print("\nDone.")


if __name__ == "__main__":
    main()

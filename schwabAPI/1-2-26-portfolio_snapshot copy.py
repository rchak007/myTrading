import os
from datetime import datetime

import pandas as pd
import dotenv
import schwabdev


def make_client():
    """Create Schwabdev client using env vars from .env."""
    dotenv.load_dotenv()

    app_key = os.getenv("app_key")
    app_secret = os.getenv("app_secret")
    callback_url = os.getenv("callback_url")

    if not app_key or not app_secret or not callback_url:
        raise RuntimeError(
            "Missing app_key, app_secret, or callback_url in .env"
        )

    # Schwabdev client – this is the same pattern you already used
    client = schwabdev.Client(app_key, app_secret, callback_url)
    return client


def get_account_balances(client) -> pd.DataFrame:
    """
    Returns one row per account: account_number, type, equity, cash, long_market_value.
    Uses account_details_all(), which you already confirmed works.
    """
    resp = client.account_details_all()
    data = resp.json()

    rows = []
    for acct in data:
        sa = acct.get("securitiesAccount", {})
        balances = sa.get("currentBalances", {}) or {}

        account_number = sa.get("accountNumber")
        account_type = sa.get("type")

        equity = balances.get("equity", balances.get("liquidationValue", 0.0))
        cash = (
            balances.get("cashBalance")
            or balances.get("cashAvailableForTrading")
            or 0.0
        )
        long_mv = balances.get("longMarketValue", 0.0)

        rows.append(
            {
                "account_number": account_number,
                "account_type": account_type,
                "equity": float(equity or 0.0),
                "cash": float(cash or 0.0),
                "long_market_value": float(long_mv or 0.0),
            }
        )

    return pd.DataFrame(rows)


def get_positions_all_accounts(client) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - positions_df: one row per position per account
      - by_symbol_df: aggregated across all accounts, per symbol
    """
    # Ask Schwab for balances + positions in one call
    try:
        resp = client.account_details_all(fields="positions")
    except TypeError:
        # Some versions of the library expect a list of fields instead of a string
        resp = client.account_details_all(fields=["positions"])

    data = resp.json()

    rows = []

    for acct in data:
        sa = acct.get("securitiesAccount", {})
        acct_num = sa.get("accountNumber")
        acct_type = sa.get("type")
        acct_name = sa.get("accountName") or str(acct_num)

        # positions will only be present if we requested fields="positions"
        for pos in sa.get("positions", []) or []:
            instrument = pos.get("instrument", {})
            symbol = instrument.get("symbol")
            asset_type = instrument.get("assetType")

            long_qty = float(pos.get("longQuantity") or 0.0)
            short_qty = float(pos.get("shortQuantity") or 0.0)
            avg_price = float(pos.get("averagePrice") or 0.0)
            mkt_value = float(pos.get("marketValue") or 0.0)

            qty = long_qty - short_qty  # net quantity
            cost_basis = qty * avg_price
            unrealized_pl = mkt_value - cost_basis

            rows.append(
                {
                    "account_number": acct_num,
                    "account_type": acct_type,
                    "account_name": acct_name,
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "quantity": qty,
                    "avg_price": avg_price,
                    "market_value": mkt_value,
                    "cost_basis": cost_basis,
                    "unrealized_pl": unrealized_pl,
                }
            )

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    positions_df = pd.DataFrame(rows)

    # Aggregate across all accounts by symbol
    by_symbol = (
        positions_df.groupby("symbol", as_index=False)
        .agg(
            {
                "quantity": "sum",
                "market_value": "sum",
                "cost_basis": "sum",
            }
        )
    )

    by_symbol["unrealized_pl"] = (
        by_symbol["market_value"] - by_symbol["cost_basis"]
    )

    # Avoid divide-by-zero
    cost_nonzero = by_symbol["cost_basis"].replace(0, pd.NA)
    by_symbol["pl_pct"] = (by_symbol["unrealized_pl"] / cost_nonzero) * 100.0

    return positions_df, by_symbol



def main():
    print("=== Schwab Portfolio Snapshot ===")
    client = make_client()

    print("\nFetching account balances...")
    df_accts = get_account_balances(client)

    print("\n=== Account Equity Snapshot ===")
    if df_accts.empty:
        print("No accounts returned.")
    else:
        print(df_accts.to_string(index=False))

        total_equity = df_accts["equity"].sum()
        total_cash = df_accts["cash"].sum()
        print("\nTotal equity across all accounts: ${:,.2f}".format(total_equity))
        print("Total cash across all accounts:   ${:,.2f}".format(total_cash))

    print("\nFetching positions and building P&L by symbol...")
    positions_df, by_symbol = get_positions_all_accounts(client)

    if positions_df.empty:
        print("\nNo positions found.")
    else:
        print("\n=== Sample Positions (first 10 rows) ===")
        print(
            positions_df.head(10).to_string(
                index=False,
                columns=[
                    "account_number",
                    "symbol",
                    "quantity",
                    "avg_price",
                    "market_value",
                    "cost_basis",
                    "unrealized_pl",
                ],
            )
        )

    if not by_symbol.empty:
        print("\n=== Aggregated P&L by Symbol (Top 20 by market value) ===")
        top = by_symbol.sort_values("market_value", ascending=False).head(20)
        print(
            top.to_string(
                index=False,
                columns=[
                    "symbol",
                    "quantity",
                    "market_value",
                    "cost_basis",
                    "unrealized_pl",
                    "pl_pct",
                ],
            )
        )

    print("\nSnapshot taken at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()

import os

import dotenv
import pandas as pd
import streamlit as st
import schwabdev


# ---------------------------------------------------------
# Client setup
# ---------------------------------------------------------

dotenv.load_dotenv()

APP_KEY = os.getenv("app_key")
APP_SECRET = os.getenv("app_secret")
CALLBACK_URL = os.getenv("callback_url")

if not APP_KEY or not APP_SECRET or not CALLBACK_URL:
    raise RuntimeError("Set app_key, app_secret, callback_url in your .env")


@st.cache_resource
def get_client():
    """Cached Schwabdev client for the Streamlit app."""
    client = schwabdev.Client(APP_KEY, APP_SECRET, CALLBACK_URL)
    return client


# ---------------------------------------------------------
# Data helpers (same logic as portfolio_snapshot, but reused)
# ---------------------------------------------------------

def get_account_balances(client) -> pd.DataFrame:
    resp = client.account_details_all()
    data = resp.json()

    rows = []
    for acct in data:
        sa = acct.get("securitiesAccount", {})
        balances = sa.get("currentBalances", {}) or {}

        account_number = sa.get("accountNumber")
        account_type = sa.get("type")
        account_name = sa.get("accountNickname") or account_number

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
                "account_name": account_name,
                "account_type": account_type,
                "equity": float(equity or 0.0),
                "cash": float(cash or 0.0),
                "long_market_value": float(long_mv or 0.0),
            }
        )

    return pd.DataFrame(rows)

def get_positions_all_accounts(client) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        resp = client.account_details_all(fields="positions")
    except TypeError:
        resp = client.account_details_all(fields=["positions"])

    data = resp.json()
    rows = []

    for acct in data:
        sa = acct.get("securitiesAccount", {})
        acct_num = sa.get("accountNumber")
        acct_type = sa.get("type")
        acct_name = sa.get("accountNickname") or acct_num

        for pos in sa.get("positions", []) or []:
            inst = pos.get("instrument", {})
            symbol = inst.get("symbol")
            description = inst.get("description")
            asset_type = inst.get("assetType")

            long_qty = float(pos.get("longQuantity", 0.0) or 0.0)
            short_qty = float(pos.get("shortQuantity", 0.0) or 0.0)
            qty = long_qty - short_qty

            avg_price = float(pos.get("averagePrice", 0.0) or 0.0)
            mkt_value = float(pos.get("marketValue", 0.0) or 0.0)

            cost_basis = qty * avg_price
            unrealized_pl = mkt_value - cost_basis

            rows.append(
                {
                    "account_number": acct_num,
                    "account_type": acct_type,
                    "account_name": acct_name,
                    "symbol": symbol,
                    "description": description,
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

    by_symbol = (
        positions_df.groupby("symbol", as_index=False)
        .agg(
            quantity=("quantity", "sum"),
            market_value=("market_value", "sum"),
            cost_basis=("cost_basis", "sum"),
        )
    )
    by_symbol["unrealized_pl"] = (
        by_symbol["market_value"] - by_symbol["cost_basis"]
    )
    cost_nonzero = by_symbol["cost_basis"].replace(0, pd.NA)
    by_symbol["pl_pct"] = (by_symbol["unrealized_pl"] / cost_nonzero) * 100.0

    return positions_df, by_symbol


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

def main():
    st.set_page_config(page_title="Schwab Portfolio Dashboard", layout="wide")
    st.title("📈 Schwab Portfolio Dashboard (All Accounts)")

    client = get_client()

    tab_accts, tab_symbols = st.tabs(["Accounts", "Symbols"])

    # ---------- Accounts tab ----------
    with tab_accts:
        st.subheader("Account Summary")

        df_accts = get_account_balances(client)

        if df_accts.empty:
            st.warning("No accounts returned from Schwab.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                total_equity = df_accts["equity"].sum()
                st.metric("Total Equity (All Accounts)", f"${total_equity:,.2f}")
            with col2:
                total_cash = df_accts["cash"].sum()
                st.metric("Total Cash (All Accounts)", f"${total_cash:,.2f}")

            st.dataframe(df_accts, use_container_width=True)

    # ---------- Symbols tab ----------
    with tab_symbols:
        st.subheader("Positions & P/L by Symbol (All Accounts Combined)")

        positions_df, by_symbol = get_positions_all_accounts(client)

        if by_symbol.empty:
            st.info("No positions found.")
        else:
            sort_option = st.selectbox(
                "Sort by:",
                ["market_value", "unrealized_pl", "pl_pct"],
                index=0,
            )
            asc = st.checkbox("Sort ascending?", value=False)

            st.markdown("### Aggregated by Symbol")
            st.dataframe(
                by_symbol.sort_values(sort_option, ascending=asc),
                use_container_width=True,
            )

            st.markdown("### Raw Positions (per account)")
            # Optional filter: by account
            acct_list = sorted(positions_df["account_number"].unique())
            selected_acct = st.selectbox(
                "Filter positions by account (optional):",
                options=["ALL"] + acct_list,
            )
            if selected_acct != "ALL":
                filtered = positions_df[
                    positions_df["account_number"] == selected_acct
                ]
            else:
                filtered = positions_df

            st.dataframe(
                filtered[
                    [
                        "account_number",
                        "symbol",
                        "description",
                        "quantity",
                        "avg_price",
                        "market_value",
                        "cost_basis",
                        "unrealized_pl",
                    ]
                ],
                use_container_width=True,
            )


if __name__ == "__main__":
    main()

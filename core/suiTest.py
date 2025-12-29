import requests
from decimal import Decimal, ROUND_DOWN, getcontext
import yfinance as yf

getcontext().prec = 50

# ---------------- CONFIG ----------------

SUI_RPC = "https://fullnode.mainnet.sui.io:443"
DEEPBOOK_INDEXER = "https://deepbook-indexer.mainnet.mystenlabs.com"

WALLET = "0x246c7037d5fd8c424e45631b930c2f3acacbee27a07a6863e797b8700a6f331d"

DEEP_COIN_TYPE = "0xdeeb7a4662eec9f2f3def03fb937a663dddaa2e215b8078a284d026b7946c270::deep::DEEP"
USDC_COIN_TYPE = "0xdba34672e30cb065b1f93e3ab55318768fd6fef66c15942c9f7cb846e2f900e7::usdc::USDC"
SUI_COIN_TYPE  = "0x2::sui::SUI"

PAIR_NAME = "DEEP_USDC"
YAHOO_TICKER_SUI = "SUI20947-USD"

# ---------------------------------------


def rpc_call(method: str, params: list):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    r = requests.post(SUI_RPC, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data["result"]


def fmt4(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def get_coin_metadata(coin_type: str):
    return rpc_call("suix_getCoinMetadata", [coin_type])


def get_balance(owner: str, coin_type: str):
    return rpc_call("suix_getBalance", [owner, coin_type])


def to_ui_amount(raw: str, decimals: int) -> Decimal:
    return Decimal(raw) / (Decimal(10) ** decimals)


def get_deep_price_usdc() -> Decimal:
    r = requests.get(f"{DEEPBOOK_INDEXER}/summary", timeout=30)
    r.raise_for_status()
    for row in r.json():
        if row.get("trading_pairs") == PAIR_NAME:
            return Decimal(str(row["last_price"]))
    raise RuntimeError("DEEP_USDC price not found")


def get_yahoo_price_usd(ticker: str) -> Decimal:
    t = yf.Ticker(ticker)

    # Prefer fast_info when available
    try:
        fi = getattr(t, "fast_info", None)
        if fi and fi.get("last_price") is not None:
            return Decimal(str(fi["last_price"]))
    except Exception:
        pass

    hist = t.history(period="5d", interval="1d")
    if hist is None or hist.empty:
        raise RuntimeError(f"No Yahoo price data returned for {ticker}")

    return Decimal(str(hist["Close"].dropna().iloc[-1]))


def main():
    # ---- metadata ----
    deep_meta = get_coin_metadata(DEEP_COIN_TYPE)
    usdc_meta = get_coin_metadata(USDC_COIN_TYPE)
    sui_meta  = get_coin_metadata(SUI_COIN_TYPE)

    # ---- balances ----
    deep_raw = get_balance(WALLET, DEEP_COIN_TYPE)
    usdc_raw = get_balance(WALLET, USDC_COIN_TYPE)
    sui_raw  = get_balance(WALLET, SUI_COIN_TYPE)

    deep_qty = to_ui_amount(deep_raw["totalBalance"], int(deep_meta["decimals"]))
    usdc_qty = to_ui_amount(usdc_raw["totalBalance"], int(usdc_meta["decimals"]))
    sui_qty  = to_ui_amount(sui_raw["totalBalance"],  int(sui_meta["decimals"]))

    # ---- prices ----
    deep_price = get_deep_price_usdc()                  # USDC per DEEP
    sui_price  = get_yahoo_price_usd(YAHOO_TICKER_SUI)  # USD per SUI

    # ---- USD values ----
    deep_usd = deep_qty * deep_price
    sui_usd  = sui_qty * sui_price

    # ---- output ----
    print(f"Wallet: {WALLET} (Sui)")
    print("")
    print(f"DEEP quantity: {fmt4(deep_qty)}")
    print(f"DEEP price (USDC): {fmt4(deep_price)}")
    print(f"DEEP value (USD): {fmt4(deep_usd)}")
    print("")
    print(f"USDC quantity: {fmt4(usdc_qty)}")
    print(f"USDC value (USD): {fmt4(usdc_qty)}")
    print("")
    print(f"SUI quantity: {fmt4(sui_qty)}")
    print(f"SUI price (USD) [{YAHOO_TICKER_SUI}]: {fmt4(sui_price)}")
    print(f"SUI value (USD): {fmt4(sui_usd)}")


if __name__ == "__main__":
    main()

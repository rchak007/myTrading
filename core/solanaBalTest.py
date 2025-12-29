import requests
# from decimal import Decimal
from decimal import Decimal, getcontext, ROUND_DOWN
import yfinance as yf

getcontext().prec = 28  # good precision for token math

RPC_URL = "https://api.mainnet-beta.solana.com"

WALLET = "315aEJ995exQtYuPdxr3EHWn9F8Pd2kciXhVQAePU8PC"

USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
MON_MINT  = "CrAr4RRJMBVwRsZtT62pEhfA9H5utymC2mVx8e7FreP2"

YAHOO_TICKER_MON = "MON30495-USD"


def rpc_call(method: str, params: list):
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }
    r = requests.post(RPC_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")
    return data["result"]


def get_spl_token_balance(wallet_pubkey: str, mint: str) -> Decimal:
    """
    Returns SPL token balance for `mint` in `wallet_pubkey` as a Decimal.
    If wallet has no associated token account for that mint, returns 0.
    """
    # Get all token accounts owned by wallet for this mint
    result = rpc_call(
        "getTokenAccountsByOwner",
        [
            wallet_pubkey,
            {"mint": mint},
            {"encoding": "jsonParsed"},
        ],
    )

    value = result.get("value", [])
    if not value:
        return Decimal("0")

    total = Decimal("0")
    for acct in value:
        parsed = acct["account"]["data"]["parsed"]["info"]
        token_amount = parsed["tokenAmount"]  # contains amount, decimals, uiAmountString
        ui = token_amount.get("uiAmountString")
        if ui is None:
            # fallback: amount is raw integer string, decimals is int
            raw = Decimal(token_amount["amount"])
            decimals = int(token_amount["decimals"])
            ui = str(raw / (Decimal(10) ** decimals))
        total += Decimal(ui)

    return total


def get_yahoo_last_price_usd(ticker: str) -> Decimal:
    """
    Gets the most recent available close/last price from Yahoo via yfinance.
    """
    t = yf.Ticker(ticker)

    # Prefer fast_info when available, otherwise fall back to recent history
    price = None
    try:
        fi = getattr(t, "fast_info", None)
        if fi and "last_price" in fi and fi["last_price"] is not None:
            price = fi["last_price"]
    except Exception:
        pass

    if price is None:
        hist = t.history(period="5d", interval="1d")
        if hist is None or hist.empty:
            raise RuntimeError(f"No price data returned from Yahoo for {ticker}")
        price = float(hist["Close"].dropna().iloc[-1])

    return Decimal(str(price))

def fmt4(x: Decimal) -> Decimal:
    """
    Format Decimal to exactly 4 decimal places (no rounding up).
    """
    return x.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)

def main():
    usdc_bal = get_spl_token_balance(WALLET, USDC_MINT)
    mon_bal  = get_spl_token_balance(WALLET, MON_MINT)

    mon_price = get_yahoo_last_price_usd(YAHOO_TICKER_MON)
    mon_usd_value = (mon_bal * mon_price)

    # print(f"Wallet: {WALLET}")
    # print(f"USDC balance: {usdc_bal}")
    # print(f"MON  balance: {mon_bal}")
    # print(f"MON price (USD) [{YAHOO_TICKER_MON}]: {mon_price}")
    # print(f"MON balance USD: {mon_usd_value}")

    print(f"Wallet: {WALLET}")

    print(f"USDC balance: {fmt4(usdc_bal)}")

    print(f"MON balance: {fmt4(mon_bal)}")
    print(f"MON price (USD) [{YAHOO_TICKER_MON}]: {fmt4(mon_price)}")
    print(f"MON balance USD: {fmt4(mon_usd_value)}")    

if __name__ == "__main__":
    main()

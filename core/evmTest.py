import time
import requests
from decimal import Decimal, getcontext, ROUND_DOWN
import yfinance as yf

getcontext().prec = 28

# ---------------- CONFIG ----------------

RPC_URLS = [
    "https://eth.llamarpc.com",
    "https://ethereum.publicnode.com",
]

WALLET = "0x6A30aA8E9Ae3Ed24c565Bf1f4060Ab167b0DA042"

PNK_CONTRACT  = "0x93ed3fbe21207ec2e8f2d3c3de6e058cb73bc04d"
USDC_CONTRACT = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"

YAHOO_TICKER_PNK = "PNK-USD"

# ERC20 method selectors
BALANCE_OF_SIG = "0x70a08231"  # balanceOf(address)
DECIMALS_SIG   = "0x313ce567"  # decimals()

# ----------------------------------------


def fmt4(x: Decimal) -> str:
    return str(x.quantize(Decimal("0.0001"), rounding=ROUND_DOWN))


def pad_address(addr: str) -> str:
    return addr.lower().replace("0x", "").rjust(64, "0")


def rpc_call(method: str, params: list, tries_per_rpc: int = 2, timeout: int = 30):
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

    last_err = None
    for url in RPC_URLS:
        for attempt in range(tries_per_rpc):
            try:
                r = requests.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                data = r.json()

                if "error" in data:
                    last_err = RuntimeError(f"RPC error from {url}: {data['error']}")
                    time.sleep(0.25 * (attempt + 1))
                    continue

                return data["result"]

            except Exception as e:
                last_err = RuntimeError(f"RPC failure from {url}: {e}")
                time.sleep(0.25 * (attempt + 1))

    raise last_err if last_err else RuntimeError("Unknown RPC failure")


def get_erc20_decimals(contract: str) -> int:
    result = rpc_call("eth_call", [{"to": contract, "data": DECIMALS_SIG}, "latest"])
    return int(result, 16)


def get_erc20_balance(wallet: str, contract: str) -> Decimal:
    data = BALANCE_OF_SIG + pad_address(wallet)
    raw = rpc_call("eth_call", [{"to": contract, "data": data}, "latest"])
    balance_int = Decimal(int(raw, 16))
    decimals = get_erc20_decimals(contract)
    return balance_int / (Decimal(10) ** decimals)


def get_yahoo_price(ticker: str) -> Decimal:
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
    usdc_bal = get_erc20_balance(WALLET, USDC_CONTRACT)
    pnk_bal  = get_erc20_balance(WALLET, PNK_CONTRACT)

    pnk_price = get_yahoo_price(YAHOO_TICKER_PNK)
    pnk_usd   = pnk_bal * pnk_price

    print(f"Wallet: {WALLET}")
    print(f"USDC balance: {fmt4(usdc_bal)}")

    print(f"PNK balance: {fmt4(pnk_bal)}")
    print(f"PNK price (USD) [{YAHOO_TICKER_PNK}]: {fmt4(pnk_price)}")
    print(f"PNK balance USD: {fmt4(pnk_usd)}")


if __name__ == "__main__":
    main()

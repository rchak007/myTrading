import time
import requests
from decimal import Decimal, getcontext, ROUND_DOWN

getcontext().prec = 28

# ---------------- CONFIG ----------------

BASE_RPC_URLS = [
    "https://mainnet.base.org",
    "https://base.publicnode.com",
]

WALLET = "0xAb7f2cdec9a5706C0c6790dd7A46db16Be59293b"

VIRTUAL_CONTRACT = "0x0b3e328455c4059EEb9e3f84b5543F74E24e7E1b"
USDC_CONTRACT    = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

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
    for url in BASE_RPC_URLS:
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


def main():
    usdc_bal = get_erc20_balance(WALLET, USDC_CONTRACT)
    virt_bal = get_erc20_balance(WALLET, VIRTUAL_CONTRACT)

    print(f"Wallet: {WALLET} (Base)")
    print(f"USDC balance: {fmt4(usdc_bal)}")
    print(f"VIRTUAL balance: {fmt4(virt_bal)}")


if __name__ == "__main__":
    main()

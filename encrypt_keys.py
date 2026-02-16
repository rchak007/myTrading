#!/usr/bin/env python3
"""
encrypt_keys.py — Fernet key‑encryption utility for jupBot wallets.

Usage:
  # 1) First time: generate a Fernet key file
  python encrypt_keys.py genkey

  # 2) Encrypt a private key (interactive prompt, hidden input)
  python encrypt_keys.py encrypt

  # 3) Encrypt with a custom key‑file path
  python encrypt_keys.py encrypt --keyfile /path/to/my.key

  # 4) Verify: decrypt an encrypted blob to confirm it round‑trips
  python encrypt_keys.py verify

The encrypted string it prints is what you paste into your .env as:
  SOLANA_PRIVATE_KEY_ENC=<encrypted_string>

The Fernet key file is what your bot reads at runtime via JUPBOT_FERNET_KEY_PATH.
"""

import argparse
import getpass
import os
import stat
import sys

from cryptography.fernet import Fernet, InvalidToken

DEFAULT_KEY_PATH = "/etc/myTrading/jupbot.key"


# ── Generate ────────────────────────────────────────────────────────
def cmd_genkey(args):
    path = args.keyfile
    if os.path.exists(path):
        print(f"⚠️  Key file already exists: {path}")
        ans = input("Overwrite? (y/N): ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "wb") as f:
        f.write(key)

    # Lock down permissions: owner read‑only
    os.chmod(path, stat.S_IRUSR)

    print(f"✅ Fernet key saved to: {path}")
    print(f"   Permissions set to 0400 (owner read‑only).")
    print(f"   Keep this file safe — anyone with it can decrypt your keys.")


# ── Encrypt ─────────────────────────────────────────────────────────
def cmd_encrypt(args):
    path = args.keyfile
    if not os.path.exists(path):
        print(f"❌ Key file not found: {path}")
        print(f"   Run:  python encrypt_keys.py genkey --keyfile {path}")
        sys.exit(1)

    with open(path, "rb") as f:
        fernet_key = f.read().strip()

    f = Fernet(fernet_key)

    label = input("Label (e.g. wallet name, optional): ").strip()
    pk = getpass.getpass("Paste base58 private key (hidden): ").strip()

    if not pk:
        print("❌ Empty key. Aborted.")
        sys.exit(1)

    encrypted = f.encrypt(pk.encode("utf-8")).decode("utf-8")

    print()
    if label:
        print(f"# {label}")
    print(f"SOLANA_PRIVATE_KEY_ENC={encrypted}")
    print()
    print("👆 Paste that line into your .env file.")
    print("   (The old plaintext SOLANA_PRIVATE_KEY_B58 line can be removed.)")


# ── Verify ──────────────────────────────────────────────────────────
def cmd_verify(args):
    path = args.keyfile
    if not os.path.exists(path):
        print(f"❌ Key file not found: {path}")
        sys.exit(1)

    with open(path, "rb") as f:
        fernet_key = f.read().strip()

    fernet = Fernet(fernet_key)

    blob = getpass.getpass("Paste encrypted blob (hidden): ").strip()

    try:
        decrypted = fernet.decrypt(blob.encode("utf-8")).decode("utf-8")
        # Show only first/last 4 chars for safety
        masked = decrypted[:4] + "..." + decrypted[-4:] if len(decrypted) > 8 else "***"
        print(f"✅ Decryption successful!  Key preview: {masked}")
    except InvalidToken:
        print("❌ Decryption failed — wrong key file or corrupted blob.")
        sys.exit(1)


# ── Bulk encrypt (for scaling to many wallets) ─────────────────────
def cmd_bulk(args):
    path = args.keyfile
    if not os.path.exists(path):
        print(f"❌ Key file not found: {path}")
        sys.exit(1)

    with open(path, "rb") as f:
        fernet_key = f.read().strip()

    f = Fernet(fernet_key)

    print("Bulk encrypt mode. Enter wallet entries one by one.")
    print("Type 'done' when finished.\n")

    results = []
    i = 1
    while True:
        label = input(f"Wallet #{i} label (or 'done'): ").strip()
        if label.lower() == "done":
            break
        pk = getpass.getpass(f"  Private key for '{label}' (hidden): ").strip()
        if not pk:
            print("  Skipped (empty).")
            continue

        encrypted = f.encrypt(pk.encode("utf-8")).decode("utf-8")
        env_var = f"SOLANA_PK_{label.upper().replace(' ', '_').replace('-', '_')}"
        results.append((label, env_var, encrypted))
        i += 1

    if results:
        print("\n" + "=" * 60)
        print("Add these to your .env file(s):\n")
        for label, var, enc in results:
            print(f"# {label}")
            print(f"{var}={enc}")
            print()
        print("=" * 60)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fernet encryption utility for Solana wallet keys"
    )
    parser.add_argument(
        "--keyfile", default=DEFAULT_KEY_PATH,
        help=f"Path to Fernet key file (default: {DEFAULT_KEY_PATH})"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("genkey",  help="Generate a new Fernet key file")
    sub.add_parser("encrypt", help="Encrypt a single private key")
    sub.add_parser("verify",  help="Verify an encrypted blob decrypts correctly")
    sub.add_parser("bulk",    help="Encrypt multiple wallet keys at once")

    args = parser.parse_args()

    cmds = {
        "genkey":  cmd_genkey,
        "encrypt": cmd_encrypt,
        "verify":  cmd_verify,
        "bulk":    cmd_bulk,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
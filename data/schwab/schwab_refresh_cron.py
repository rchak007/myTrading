# schwab_refresh_cron.py
import os
from pathlib import Path
from data.schwab.schwab_helper import create_schwab_client

TOKEN_PATHS = [Path("tokens.json")]
USER_ID = "main_local"

try:
    client = create_schwab_client(USER_ID, TOKEN_PATHS)
    client.get_client()  # This auto-refreshes if needed
    print("✅ Tokens refreshed successfully")
except Exception as e:
    print(f"❌ Refresh failed: {e}")
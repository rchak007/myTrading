# data/schwab/schwab_helper.py
"""
Schwab authentication helper with proper token refresh handling.
"""
from __future__ import annotations

import os
import schwabdev
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from data.schwab.token_store import load_tokens_db, save_tokens_db
from data.schwab.token_sync import sync_db_to_local_multi, sync_local_to_db_multi


class SchwabAuthError(Exception):
    """Raised when Schwab authentication fails and manual re-auth is needed."""
    pass


class SchwabClient:
    """
    Wrapper around schwabdev.Client that handles token refresh and syncing.
    """
    
    def __init__(
        self,
        user_id: str,
        token_paths: list[Path],
        app_key: str,
        app_secret: str,
        callback_url: str,
    ):
        self.user_id = user_id
        self.token_paths = token_paths
        self.app_key = app_key
        self.app_secret = app_secret
        self.callback_url = callback_url
        self._client: Optional[schwabdev.Client] = None
        
    def _sync_db_to_local(self) -> bool:
        """Pull tokens from DB to local files."""
        return sync_db_to_local_multi(self.user_id, self.token_paths)
    
    def _sync_local_to_db(self):
        """Push tokens from local files to DB."""
        sync_local_to_db_multi(self.user_id, self.token_paths)
    
    def _check_token_expiry(self) -> dict:
        """
        Check token status from DB.
        Returns: {"expired": bool, "needs_refresh": bool, "seconds_left": int}
        """
        db_token = load_tokens_db(self.user_id)
        if not db_token:
            return {"expired": True, "needs_refresh": True, "seconds_left": 0}
        
        now = datetime.now(timezone.utc)
        expires_at = datetime.fromtimestamp(db_token["expires_at"], tz=timezone.utc)
        seconds_left = int((expires_at - now).total_seconds())
        
        return {
            "expired": seconds_left <= 0,
            "needs_refresh": seconds_left <= 300,  # Refresh if < 5 min left
            "seconds_left": seconds_left,
        }
    
    def get_client(self, force_new_auth: bool = False) -> schwabdev.Client:
        """
        Get authenticated schwabdev client.
        Handles token refresh automatically.
        
        Args:
            force_new_auth: If True, deletes tokens and forces fresh OAuth login
        
        Raises:
            SchwabAuthError if manual re-authentication is needed.
        """
        # If forcing new auth, delete all tokens first
        if force_new_auth:
            for p in self.token_paths:
                p.unlink(missing_ok=True)
            # Don't check expiry, just proceed to create client
            pass
        else:
            # Check token status only if not forcing new auth
            status = self._check_token_expiry()
            
            if status["expired"]:
                raise SchwabAuthError(
                    "Tokens expired. Manual re-authentication required.\n"
                    "Click 'Clear BOTH (DB + Local)' then run Schwab test again."
                )
            
            # Sync DB → local before creating client
            if not self._sync_db_to_local():
                # No tokens in DB - we'll create client which will trigger OAuth
                print("⚠️  No tokens found. Starting OAuth flow...")
        
        # Point schwabdev to the first token file
        os.environ["SCHWAB_TOKEN_PATH"] = str(self.token_paths[0])
        
        # Create client (schwabdev will auto-refresh if needed OR start OAuth)
        print(f"🔐 Creating Schwab client with callback: {self.callback_url}")
        try:
            self._client = schwabdev.Client(
                self.app_key,
                self.app_secret,
                self.callback_url,
            )
            print("✅ Client created successfully!")
        except Exception as e:
            error_msg = str(e)
            # If client creation fails, it's likely a token issue
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                raise SchwabAuthError(
                    "Authentication failed (401). Tokens may be invalid.\n"
                    "Click 'Clear BOTH (DB + Local)' then re-authenticate."
                )
            # If it's asking for callback URL, re-raise with helpful message
            if "redirect" in error_msg.lower() or "callback" in error_msg.lower():
                raise SchwabAuthError(
                    f"OAuth callback issue: {error_msg}\n"
                    f"Make sure your callback URL ({self.callback_url}) is registered in Schwab Developer Portal."
                )
            raise
        
        # After client creation, sync local → DB (in case tokens were refreshed)
        self._sync_local_to_db()
        
        return self._client
    
    def fetch_positions(self) -> dict:
        """
        Fetch account positions from Schwab.
        Returns raw JSON response.
        Raises SchwabAuthError if re-authentication needed.
        """
        client = self.get_client()
        
        try:
            # Try with string first (schwabdev sometimes wants string, sometimes list)
            resp = client.account_details_all(fields="positions")
        except TypeError:
            resp = client.account_details_all(fields=["positions"])
        except Exception as e:
            # Check for auth errors
            if hasattr(resp, 'status_code') and resp.status_code in (401, 403):
                raise SchwabAuthError(
                    f"Schwab API returned {resp.status_code}. Re-authentication needed.\n"
                    "Click 'Clear BOTH (DB + Local)' then re-authenticate."
                )
            raise
        
        # Sync tokens back to DB (in case they were refreshed during the call)
        self._sync_local_to_db()
        
        if resp.status_code != 200:
            raise SchwabAuthError(
                f"Schwab API error: {resp.status_code}\n"
                f"Response: {resp.text[:500]}\n"
                "This usually means tokens are invalid. Try re-authenticating."
            )
        
        return resp.json()


def create_schwab_client(user_id: str, token_paths: list[Path]) -> SchwabClient:
    """
    Factory function to create a SchwabClient with env vars.
    """
    app_key = os.getenv("app_key")
    app_secret = os.getenv("app_secret")
    callback_url = os.getenv("callback_url")
    
    if not all([app_key, app_secret, callback_url]):
        raise ValueError(
            "Missing Schwab env vars: app_key, app_secret, or callback_url"
        )
    
    return SchwabClient(
        user_id=user_id,
        token_paths=token_paths,
        app_key=app_key,
        app_secret=app_secret,
        callback_url=callback_url,
    )
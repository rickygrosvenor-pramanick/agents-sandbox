"""
utils/jira_client.py

Simple Jira Cloud client for creating Story issues.
Reads credentials from environment variables (load .env if present):
- JIRA_BASE_URL (e.g., https://your-domain.atlassian.net)
- JIRA_EMAIL (account email)
- JIRA_API_TOKEN (API token)
- JIRA_PROJECT_KEY (default project key)

Usage:
from utils.jira_client import create_story
issue = create_story("My summary", "My description", project_key="ABC", labels=["ba","rag"])
print(issue["key"])  # e.g., ABC-123
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any

import requests
from dotenv import load_dotenv

# Load env once
load_dotenv()


def _get_jira_config() -> Dict[str, Optional[str]]:
    base_url = os.getenv("JIRA_BASE_URL", "").rstrip("/")
    email = os.getenv("JIRA_EMAIL")
    token = os.getenv("JIRA_API_TOKEN")
    default_project = os.getenv("JIRA_PROJECT_KEY")
    if not base_url:
        raise RuntimeError("JIRA_BASE_URL is not set")
    if not email or not token:
        raise RuntimeError("JIRA_EMAIL or JIRA_API_TOKEN is not set")
    return {
        "base_url": base_url,
        "email": email,
        "token": token,
        "default_project": default_project,
    }


def create_story(
    summary: str,
    description: str,
    project_key: Optional[str] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = _get_jira_config()
    base_url = cfg["base_url"]
    email = cfg["email"]
    token = cfg["token"]
    project = project_key or cfg["default_project"]
    if not project:
        raise RuntimeError("No project key provided and JIRA_PROJECT_KEY not set")

    url = f"{base_url}/rest/api/2/issue"
    auth = (email, token)
    headers = {"Content-Type": "application/json"}

    payload: Dict[str, Any] = {
        "fields": {
            "project": {"key": project},
            "summary": summary[:250],  # Jira summary max is 255 chars
            "description": description,
            "issuetype": {"name": "Story"},
        }
    }
    if labels:
        payload["fields"]["labels"] = labels

    resp = requests.post(url, json=payload, headers=headers, auth=auth, timeout=30)
    if resp.status_code not in (200, 201):
        try:
            detail = resp.json()
        except Exception:
            detail = {"text": resp.text}
        raise RuntimeError(f"Jira create story failed: {resp.status_code} {detail}")

    return resp.json()

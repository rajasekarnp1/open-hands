#!/usr/bin/env python3
"""
Upload project files to GitHub using the API
"""

import os
import base64
import json
import requests
from pathlib import Path

def upload_file_to_github(repo_owner, repo_name, file_path, content, token, branch="main"):
    """Upload a file to GitHub repository using the API."""

    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"

    # Encode content to base64
    if isinstance(content, str):
        content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    else:
        content_encoded = base64.b64encode(content).decode('utf-8')

    data = {
        "message": f"Add {file_path}",
        "content": content_encoded,
        "branch": branch
    }

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.put(url, json=data, headers=headers)

    if response.status_code in [200, 201]:
        print(f"✅ Successfully uploaded {file_path}")
        return True
    else:
        print(f"❌ Failed to upload {file_path}: {response.status_code} - {response.text}")
        return False

def upload_project_to_github():
    """Upload the entire project to GitHub."""

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        print("❌ GITHUB_TOKEN not found")
        return False

    repo_owner = "Subikshaa1910"
    repo_name = "openhands"

    # Key files to upload
    key_files = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "main.py",
        "demo.py",
        "enhanced_demo.py",
        "experimental_demo.py",
        "experimental_optimizer.py",
        "recursive_optimizer.py",
        "openhands_improver.py",
        "ai_scientist_openhands.py",
        "auto_updater_demo.py",
        "cli.py",
        "web_ui.py",
        "docker-compose.yml",
        "Dockerfile",
        "FINAL_SUMMARY.md",
        "EXPERIMENTAL_SUMMARY.md",
        "OPENHANDS_WHOLE_PROJECT_IMPROVEMENT.md",
        "RESEARCH_ENHANCEMENTS.md",
        "RECURSIVE_SELF_IMPROVEMENT.md",
        "AUTO_UPDATER_DOCUMENTATION.md",
        "EXPERIMENTAL_FEATURES.md"
    ]

    # Upload key files
    success_count = 0
    for file_path in key_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if upload_file_to_github(repo_owner, repo_name, file_path, content, token):
                success_count += 1

    # Upload src directory structure
    src_files = [
        "src/__init__.py",
        "src/models.py",
        "src/core/__init__.py",
        "src/core/aggregator.py",
        "src/core/meta_controller.py",
        "src/core/ensemble_system.py",
        "src/core/auto_updater.py",
        "src/core/browser_monitor.py",
        "src/core/account_manager.py",
        "src/core/rate_limiter.py",
        "src/core/router.py",
        "src/providers/__init__.py",
        "src/providers/base.py",
        "src/providers/openrouter.py",
        "src/providers/groq.py",
        "src/providers/cerebras.py",
        "src/api/__init__.py",
        "src/api/server.py"
    ]

    for file_path in src_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if upload_file_to_github(repo_owner, repo_name, file_path, content, token):
                success_count += 1

    # Upload config files
    config_files = [
        "config/providers.yaml",
        "config/auto_update.yaml"
    ]

    for file_path in config_files:
        if Path(file_path).exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if upload_file_to_github(repo_owner, repo_name, file_path, content, token):
                success_count += 1

    print(f"\n🎉 Upload complete! Successfully uploaded {success_count} files to {repo_owner}/{repo_name}")
    print(f"🔗 Repository: https://github.com/{repo_owner}/{repo_name}")

    return success_count > 0

if __name__ == "__main__":
    upload_project_to_github()
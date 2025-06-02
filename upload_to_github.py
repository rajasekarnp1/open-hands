#!/usr/bin/env python3
"""
Upload project files to GitHub using the API
"""

import os
import base64
import json
import requests
import time
from pathlib import Path

def upload_file_to_github(repo_owner, repo_name, file_path, content, token, branch="main"):
    """Upload a file to GitHub repository using the API."""
    
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_path}"
    
    # Check if file already exists
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Get existing file SHA if it exists
    get_response = requests.get(url, headers=headers)
    sha = None
    if get_response.status_code == 200:
        sha = get_response.json().get("sha")
    
    # Encode content to base64
    if isinstance(content, str):
        content_encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
    else:
        content_encoded = base64.b64encode(content).decode('utf-8')
    
    data = {
        "message": f"Update {file_path}" if sha else f"Add {file_path}",
        "content": content_encoded,
        "branch": branch
    }
    
    # Add SHA if file exists (for updates)
    if sha:
        data["sha"] = sha
    
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
        "EXPERIMENTAL_FEATURES.md",
        "DEPLOYMENT_GUIDE.md",
        "PROJECT_COMPLETION_SUMMARY.md",
        "PROJECT_SUMMARY.md",
        "USAGE.md",
        "upload_to_github.py",
        "test_auto_updater_integration.py"
    ]
    
    # Upload key files
    success_count = 0
    total_files = 0
    
    print(f"🚀 Starting upload to {repo_owner}/{repo_name}")
    
    for file_path in key_files:
        if Path(file_path).exists():
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if upload_file_to_github(repo_owner, repo_name, file_path, content, token):
                    success_count += 1
                
                # Rate limiting - wait between uploads
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
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
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if upload_file_to_github(repo_owner, repo_name, file_path, content, token):
                    success_count += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    # Upload config files
    config_files = [
        "config/providers.yaml",
        "config/auto_update.yaml"
    ]
    
    for file_path in config_files:
        if Path(file_path).exists():
            total_files += 1
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if upload_file_to_github(repo_owner, repo_name, file_path, content, token):
                    success_count += 1
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
        else:
            print(f"⚠️  File not found: {file_path}")
    
    print(f"\n🎉 Upload complete! Successfully uploaded {success_count}/{total_files} files to {repo_owner}/{repo_name}")
    print(f"🔗 Repository: https://github.com/{repo_owner}/{repo_name}")
    
    if success_count == total_files:
        print("✅ All files uploaded successfully!")
    elif success_count > 0:
        print(f"⚠️  Partial upload: {total_files - success_count} files failed")
    else:
        print("❌ Upload failed - no files were uploaded")
    
    return success_count > 0

if __name__ == "__main__":
    upload_project_to_github()
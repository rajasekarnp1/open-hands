import tarfile
import os
import sys
from pathlib import Path

ARCHIVE_NAME = "openhands_ai_scientist_project.tar.gz"
EXTRACT_DIR = Path("./extracted_ai_scientist_project")

def main():
    """
    Extracts the contents of the specified tar.gz archive.
    """
    archive_path = Path(ARCHIVE_NAME)

    if not archive_path.exists():
        print(f"Error: Archive file '{ARCHIVE_NAME}' not found in the current directory.")
        sys.exit(1)

    if not EXTRACT_DIR.exists():
        try:
            EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created extraction directory: '{EXTRACT_DIR}'")
        except Exception as e:
            print(f"Error creating extraction directory '{EXTRACT_DIR}': {e}")
            sys.exit(1)
    else:
        print(f"Extraction directory '{EXTRACT_DIR}' already exists. Files might be overwritten.")

    try:
        print(f"Attempting to open and extract '{ARCHIVE_NAME}'...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
            print(f"Successfully extracted archive to '{EXTRACT_DIR}'")

            print("\nExtracted top-level contents:")
            extracted_items = []
            for item in EXTRACT_DIR.iterdir():
                extracted_items.append(item.name)
                print(f"- {item.name}")

            if not extracted_items:
                print("(No top-level items found or directory is empty after extraction)")

    except tarfile.ReadError:
        print(f"Error: '{ARCHIVE_NAME}' is not a valid tar.gz file or is corrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

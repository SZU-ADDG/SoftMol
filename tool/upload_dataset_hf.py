import os
from huggingface_hub import HfApi

# Target repository on Hugging Face for the dataset
repo_id = "SZU-ADDG/ZINC-Curated"

# Local path where the dataset is located
extract_path = "dataset/ZINC-Curated"

api = HfApi()

if not os.path.exists(extract_path):
    print(f"[Error] The slice data directory cannot be found at: {extract_path}")
    exit(1)

print(f"\nVerifying/Creating remote dataset repository: {repo_id} ...")
# Establish the remote repository as a dataset
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

print(f"Start batch syncing all Arrow slices and JSON configurations inside {extract_path} to the cloud...")

try:
    # upload_folder transfers all inner nested files seamlessly
    api.upload_folder(
        folder_path=extract_path,
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"\n[√] Dataset has been completely synchronized!")
except Exception as e:
    print(f"\n[X] Synchronization exception disconnected: {e}")

print(f"Please verify on the remote endpoint: https://huggingface.co/datasets/{repo_id}")

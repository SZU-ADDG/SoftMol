import os
from huggingface_hub import HfApi

# Target repository on Hugging Face
repo_id = "SZU-ADDG/SoftMol"
api = HfApi()

# Mapping between weight files and their corresponding config files
upload_pairs = [
    ("weights/55M-epoch1-last.ckpt", "configs/model/small-50M.yaml"),
    ("weights/74M-epoch1-last.ckpt", "configs/model/small-70M.yaml"),
    ("weights/89M-epoch6-best.ckpt", "configs/model/small-89M.yaml"),
    ("weights/116M-epoch1-last.ckpt", "configs/model/small-110M.yaml"),
    ("weights/624M-epoch1-last.ckpt", "configs/model/large.yaml"),
]

print(f"==================================================")
print(f"🚀 Starting synchronization to Hugging Face repository [{repo_id}]")
print(f"==================================================")

# Create the remote model repository if it does not exist
print(f"[Stage 1] Verifying/Creating remote repository space ...")
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Iterate through the pairs and upload each
print(f"[Stage 2] Begin uploading all specified models ...")
for ckpt_path, config_path in upload_pairs:
    print(f"\n[{ckpt_path}] -> Synchronizing weight and architecture configuration")
    
    try:
        # 1. Verify and upload the weight file
        if os.path.exists(ckpt_path):
            print(f"  -> [Uploading] Weight: {ckpt_path}")
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=ckpt_path,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            print(f"  -> [Warning] Local weight file not found: {ckpt_path}. Skipped.")

        # 2. Verify and upload the config file
        if os.path.exists(config_path):
            print(f"  -> [Uploading] Config: {config_path}")
            api.upload_file(
                path_or_fileobj=config_path,
                path_in_repo=config_path,
                repo_id=repo_id,
                repo_type="model"
            )
        else:
            print(f"  -> [Warning] Local config file not found: {config_path}. Skipped.")
            
        print(f"  [√] Model successfully synchronized.")
        
    except Exception as e:
        print(f"  [X] Synchronization interrupted due to an error: {str(e)}")

print(f"\n==================================================")
print(f"🎉 All model weights and configuration files processed successfully!")
print(f"🌍 Remote Verification URL: https://huggingface.co/{repo_id}")
print(f"==================================================")

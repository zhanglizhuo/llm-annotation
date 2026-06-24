import os
import requests

HF_MIRROR = "https://hf-mirror.com"

datasets = {
    "SCB_BowTurnHead": [
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509.zip", "SCB_BowTurnHead_20250509.zip"),
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB_BowTurnHead_20250509/SCB_BowTurnHead_20250509.yaml", "SCB_BowTurnHead_20250509.yaml"),
    ],
    "SCB5_Discuss": [
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17.zip", "SCB5-Discuss-2024-9-17.zip"),
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Discuss-2024-9-17/SCB5-Discuss-2024-9-17.yaml", "SCB5-Discuss-2024-9-17.yaml"),
    ],
    "SCB5_HandriseReadWrite": [
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip", "SCB5-Handrise-Read-write-2024-9-17.zip"),
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.yaml", "SCB5-Handrise-Read-write-2024-9-17.yaml"),
    ],
    "SCB5_TeacherBehavior": [
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2.zip", "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406-2.zip"),
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406.yaml", "SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406.yaml"),
        ("datasets/wintonYF/SCB-Dataset/resolve/main/SCB5_Teacher_Behavior_Stand_BlackBoard_Sreen_20250406/countLabels4.py", "countLabels4.py"),
    ],
}

def download_file(url, out_path, timeout=300):
    """Download with resume support"""
    headers = {}
    file_size = 0
    
    if os.path.exists(out_path):
        file_size = os.path.getsize(out_path)
        headers['Range'] = f'bytes={file_size}-'
        print(f"  Resume from {file_size} bytes...")
    
    try:
        resp = requests.get(url, stream=True, headers=headers, timeout=(30, timeout))
        resp.raise_for_status()
        
        total_size = file_size + int(resp.headers.get('content-length', 0))
        
        mode = 'ab' if file_size > 0 else 'wb'
        with open(out_path, mode) as f:
            downloaded = file_size
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

for folder, files in datasets.items():
    target_dir = os.path.join("datasets_scb", folder)
    os.makedirs(target_dir, exist_ok=True)
    
    for path, fname in files:
        url = f"{HF_MIRROR}/{path}"
        out_path = os.path.join(target_dir, fname)
        
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Skip {fname} (exists)")
            continue
            
        print(f"Downloading {fname}...")
        success = download_file(url, out_path)
        if success:
            print(f"  Done: {out_path}")
        else:
            print(f"  Failed: {fname}")

print("Download finished.")

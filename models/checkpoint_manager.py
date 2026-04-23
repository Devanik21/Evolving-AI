import os
import glob
import zipfile

def export_checkpoint(weights_path: str, config_path: str, stats_path: str, output_zip: str):
    """
    Exports a trained model checkpoint as a zip archive containing weights,
    configuration, and runtime statistics.
    """
    print(f"Exporting checkpoint to {output_zip}...")
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        if os.path.exists(weights_path):
            zipf.write(weights_path, arcname="weights.json")
        if os.path.exists(config_path):
            zipf.write(config_path, arcname="config.json")
        if os.path.exists(stats_path):
            zipf.write(stats_path, arcname="stats.json")
    print("Export complete.")

def load_checkpoint(zip_path: str, extract_dir: str):
    """
    Extracts a checkpoint zip to the target directory.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Checkpoint {zip_path} not found.")

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_dir)
    print(f"Loaded checkpoint from {zip_path} into {extract_dir}.")

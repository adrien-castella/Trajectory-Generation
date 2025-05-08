import tarfile, urllib.request
from pathlib import Path
from tqdm import tqdm

def check_and_download_argoverse():
    dataset_dir = Path("datasets/argoverse_v1.1/extracted")
    expected_dirs = ['forecasting_train', 'forecasting_val', 'forecasting_test']

    if all((dataset_dir / d).exists() for d in expected_dirs):
        print("All Argoverse datasets found.")
        return

    print("Some Argoverse datasets missing. Downloading...")

    DATASETS = {
        "forecasting_train.tar.gz": "https://s3.amazonaws.com/argoverse/datasets/av1/tars/forecasting_train.tar.gz",
        "forecasting_val.tar.gz": "https://s3.amazonaws.com/argoverse/datasets/av1/tars/forecasting_val.tar.gz",
        "forecasting_test.tar.gz": "https://s3.amazonaws.com/argoverse/datasets/av1/tars/forecasting_test.tar.gz",
    }

    download_dir = Path("datasets/argoverse_v1.1")
    extract_dir = download_dir / "extracted"
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    def download_file_with_progress(filename, url):
        file_path = download_dir / filename
        if file_path.exists():
            print(f"{filename} already exists. Skipping download.")
            return file_path

        with urllib.request.urlopen(url) as response:
            total_size = int(response.getheader("Content-Length"))
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                with open(file_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))

        return file_path

    def extract_tar_gz(file_path):
        extract_target = extract_dir / file_path.stem.replace('.tar', '')
        if extract_target.exists():
            print(f"{extract_target} already exists. Skipping extraction.")
        else:
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=extract_target)
            print(f"Extracted {file_path.name} to {extract_target}.")
        return extract_target

    for filename, url in DATASETS.items():
        file_path = download_file_with_progress(filename, url)
        extract_tar_gz(file_path)

    print("All Argoverse data downloaded and extracted.")
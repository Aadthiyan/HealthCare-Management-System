import pathlib
import gdown

# Mapping of expected filenames to Google Drive file IDs
# Update the IDs below if they change in Drive
MODELS = {
    "model.pkl": "15SGrVlw-vJ2mhebgJyxYfXXvXCoSoBzB",
    "medi_model.pkl": "1f89C_aO9sxaTTe_LXKOhWRymG0NR6q7J",
}

TARGET_DIR = pathlib.Path("recommend/data/output")
TARGET_DIR.mkdir(parents=True, exist_ok=True)


def ensure_models() -> None:
    for name, file_id in MODELS.items():
        destination_path = TARGET_DIR / name
        if destination_path.exists():
            print(f"{name} present; skipping download.")
            continue
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {name} from Google Drive...")
        gdown.download(url, str(destination_path), quiet=False)


if __name__ == "__main__":
    ensure_models()



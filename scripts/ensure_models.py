import os
import pathlib
import re
import gdown


def _to_drive_url(id_or_url: str) -> str:
    # Accepts either a raw ID or a Google Drive share URL and returns a direct download URL
    if not id_or_url:
        return ""
    # uc?id=ID form
    match = re.search(r"[?&]id=([\w-]+)", id_or_url)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"
    # file/d/ID/ form
    match = re.search(r"/file/d/([\w-]+)/", id_or_url)
    if match:
        return f"https://drive.google.com/uc?id={match.group(1)}"
    # Looks like a bare ID
    if re.fullmatch(r"[\w-]{10,}", id_or_url):
        return f"https://drive.google.com/uc?id={id_or_url}"
    # Fallback: assume it's already a direct URL
    return id_or_url


# Files to ensure (path -> Drive ID or URL). Update values or set via env vars.
FILES_TO_DOWNLOAD = {
    # Doctor recommendation models (lives under recommend/data/output)
    # Note: IDs/links provided by user
    "recommend/data/output/model.pkl": os.getenv(
        "MODEL_ID",
        "https://drive.google.com/file/d/1f89C_aO9sxaTTe_LXKOhWRymG0NR6q7J/view?usp=drive_link",
    ),
    "recommend/data/output/medi_model.pkl": os.getenv(
        "MEDI_MODEL_ID",
        "https://drive.google.com/file/d/15SGrVlw-vJ2mhebgJyxYfXXvXCoSoBzB/view?usp=drive_link",
    ),
    # Medicine recommendation pickles (expected in project root by app.py)
    # Provide IDs/URLs via env or replace placeholders below
    # Only download medicine files if explicitly configured via env
    "medicine_dict.pkl": os.getenv(
        "MEDICINE_DICT_ID",
        "https://drive.google.com/file/d/1Xa_R_e_FDsnS_uF3TU1HY9qX-t-6DeMm/view?usp=drive_link",
    ),
    "similarity.pkl": os.getenv(
        "SIMILARITY_ID",
        "https://drive.google.com/file/d/1YgczwCg_BEtzLrUEdnTAE9oc4h0lMSPX/view?usp=drive_link",
    ),
}


def ensure_models() -> None:
    for rel_path, id_or_url in FILES_TO_DOWNLOAD.items():
        dst = pathlib.Path(rel_path)
        if dst.exists():
            print(f"{rel_path} present; skipping download.")
            continue
        if not id_or_url:
            # Not configured; skip silently
            continue
        url = _to_drive_url(id_or_url)
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {rel_path} from Google Drive...")
        gdown.download(url, str(dst), quiet=False)


if __name__ == "__main__":
    ensure_models()



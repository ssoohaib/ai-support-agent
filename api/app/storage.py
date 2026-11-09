# app/storage.py
# Where uploaded files are kept. A local folder for now (the guide moves this to S3 in v10).

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "uploads"     # → api/uploads/


def save(key: str, data: bytes) -> str:
    """Save the bytes under a name like 'returns/1042/<random id>' and return where they went."""
    path = ROOT / key
    path.parent.mkdir(parents=True, exist_ok=True)              # create the folders if they don't exist yet
    path.write_bytes(data)
    return str(path)

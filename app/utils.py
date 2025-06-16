import base64
from pathlib import Path


def to_base64(path: Path) -> str:
    data = path.read_bytes()
    mime = f"image/{'jpeg' if path.suffix.lower() in {'.jpg', '.jpeg'} else path.suffix.lstrip('.')}"
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"

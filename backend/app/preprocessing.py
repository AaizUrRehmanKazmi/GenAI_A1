"""Transport validation only. Model normalization remains a research decision."""
from io import BytesIO
import warnings
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, UploadFile

MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000

async def validate_image(file: UploadFile) -> bytes:
    payload = await file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 10 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as im:
                if im.format not in {"PNG", "JPEG"}:
                    raise HTTPException(415, "Use a PNG or JPEG image.")
                if im.width * im.height > MAX_PIXELS:
                    raise HTTPException(413, "Image exceeds 20 million pixels.")
                im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(415, "Invalid or unsafe image.") from None
    return payload

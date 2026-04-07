import base64
from datetime import timezone, datetime
from io import BytesIO

from PIL import Image, ImageFilter
from firebase_functions import https_fn


def decode_base64_image(base64_image: str) -> bytes:
    if "," in base64_image:
        base64_string = base64_image.split(",")[1]
    else:
        base64_string = base64_image

    # Decode the string
    return base64.b64decode(base64_string)


def generate_image_filename(content_type: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    ext = content_type.split("/")[1]
    return f"aurafit-{timestamp}.{ext}"


def generate_thumbnail(
    image_data: bytes,
    size: tuple[float, float],
    blur: bool = True,
    blur_radius: int = 5,
    ret_as_bytes: bool = False,
) -> str | bytes:
    image = Image.open(BytesIO(image_data)).convert("RGB")
    if image.size[0] >= size[0] and image.size[1] >= size[1]:
        image.thumbnail(size)
        if blur:
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    if ret_as_bytes:
        return buffered.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


def get_dominant_color(image_data: bytes) -> str:
    image = Image.open(BytesIO(image_data))
    image = image.resize((1, 1))
    r, g, b = image.getpixel((0, 0))
    return f"#{r:02x}{g:02x}{b:02x}"


# Generate embedding for a search query
def prepare_query(query):
    return f"task: search result | query: {query}"


# Generate embedding for a search document
def prepare_document(content, title=None):
    if title is None:
        title = "none"
    return f"title: {title} | text: {content}"


# Generate embedding for classification
def prepare_classification_input(content):
    return f"task: classification | query: {content}"


def require_text_field(value: object, name: str, max_len: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"{name} is required",
        )
    val = value.strip()
    if len(val) > max_len:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"{name} is too long",
        )
    return val

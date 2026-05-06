import base64
from datetime import timezone, datetime
from io import BytesIO

from PIL import Image, ImageFilter
from firebase_admin import auth, firestore
from firebase_functions import https_fn
from google.cloud.firestore_v1 import Increment


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


def parse_media(data, user_prefix, field_name: str) -> tuple[str, str]:
    media_obj = data.get(field_name)
    if not isinstance(media_obj, dict):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"{field_name} must be an object",
        )
    gs_url = media_obj.get("gs_url")
    mimetype = media_obj.get("mimetype")
    if not isinstance(gs_url, str) or not gs_url.startswith("gs://"):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"{field_name}.gs_url must be a valid gs:// URL",
        )
    if not gs_url.startswith(user_prefix):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.PERMISSION_DENIED,
            message=f"{field_name} must belong to the authenticated user",
        )
    if not isinstance(mimetype, str) or not mimetype.startswith("image/"):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message=f"{field_name}.mimetype must be an image MIME type",
        )
    return gs_url, mimetype


def process_subscription(service, data, config):
    notification = data["subscriptionNotification"]
    token = notification["purchaseToken"]

    # Use subscriptionsv2 to get details
    package_name = data["packageName"]
    sub = (
        service.purchases()
        .subscriptionsv2()
        .get(packageName=package_name, token=token)
        .execute()
    )

    # Check if it needs acknowledgement

    if sub.get("acknowledgementState") == "ACKNOWLEDGEMENT_STATE_PENDING":
        service.purchases().subscriptions().acknowledge(
            packageName=package_name,
            subscriptionId=notification["subscriptionId"],
            token=token,
            body={},
        ).execute()
        for line in sub.get("lineItems"):
            coins = config.get_int(line["offerDetails"]["basePlanId"])
            uid = sub["externalAccountIdentifiers"]["obfuscatedExternalAccountId"]
            auth.get_user(uid)
            award_coins(uid, coins)


def process_one_time_product(service, notification, package_name):
    token = notification.get("purchaseToken")
    product_id = notification.get("sku")  # The SKU/Product ID

    # Use productsv2 to get details (no productId required for the GET)
    purchase = (
        service.purchases()
        .productsv2()
        .getproductpurchasev2(packageName=package_name, token=token)
        .execute()
    )

    # Check if it needs acknowledgement
    if purchase.get("acknowledgementState") == "ACKNOWLEDGEMENT_STATE_PENDING":
        # LOGIC: If it's a consumable (like coins), we CONSUME.
        # If it's a permanent upgrade, we ACKNOWLEDGE.

        # Example: If the product ID contains 'coins', we consume it
        if "coins" in product_id:
            service.purchases().products().consume(
                packageName=package_name, productId=product_id, token=token
            ).execute()
            print(f"Consumable {product_id} consumed.")

            # Award coins in Firestore
            user_id = purchase.get("externalAccountIdentifiers", {}).get(
                "externalAccountId"
            )
            if user_id:
                award_coins(user_id, 100)  # Example amount
        else:
            # Permanent purchase
            service.purchases().products().acknowledge(
                packageName=package_name, productId=product_id, token=token, body={}
            ).execute()
            print(f"Permanent product {product_id} acknowledged.")


def award_coins(uid, amount):
    db = firestore.client()
    db.document(f"users/{uid}").update({"coins": Increment(amount)})
    print(f"award_coins: Awarded {amount} coins to {uid}")

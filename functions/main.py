# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
import logging
import pathlib
from typing import Any

from firebase_admin import initialize_app, storage, firestore, auth
from firebase_admin.auth import UserNotFoundError
from firebase_functions import https_fn, storage_fn
from firebase_functions.options import set_global_options, MemoryOption
from genkit import Part, TextPart, MediaPart, Media, Document
from genkit.core.typing import DocumentPart
from google.cloud import firestore as cloud_firestore

from models import (
    GenerateImageInputSchema,
    ImageGenerationResult,
    ImageCategorizationInput,
    ImageCategorizationOutput,
    IndexData,
)
from prompts import (
    generate_image_system_prompt,
    categorization_system_prompt,
    categorization_user_prompt,
    generate_image_user_prompt,
)
from runflow import (
    run_generate_instantfit_flow,
    run_categorize_image_flow,
    run_generate_autofit_flow,
)
from runindexer import run_indexer
from utils import (
    generate_image_filename,
    generate_thumbnail,
    get_dominant_color,
    decode_base64_image,
    prepare_document,
    require_text_field,
)

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-function
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=10)
initialize_app()
logger = logging.getLogger(__name__)


@https_fn.on_call(timeout_sec=600, memory=MemoryOption.MB_512, cpu=2)  # type: ignore
def generate_image(req: https_fn.CallableRequest) -> dict:
    uid = getattr(req.auth, "uid", None)
    if not uid:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.UNAUTHENTICATED,
            message="You're not allowed here",
        )

    if not isinstance(req.data, dict):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Request data must be an object",
        )

    data: dict[str, Any] = req.data
    mood = require_text_field(data.get("mood"), "mood")
    occasion = require_text_field(data.get("occasion"), "occasion")
    time_of_day = require_text_field(data.get("time_of_day"), "time_of_day")

    request_type = data.get("type")
    if request_type is not None and request_type not in ("instantfit", "autofit"):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="type must be either instantfit or autofit",
        )

    db = firestore.client()
    storage_bucket = storage.bucket()
    user_prefix = f"gs://{storage_bucket.name}/users/{uid}/"

    def _parse_media(field_name: str) -> tuple[str, str]:
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

    result: ImageGenerationResult | None
    try:
        if request_type == "instantfit":
            selfie_url, selfie_mimetype = _parse_media("selfie")
            cloth_url, cloth_mimetype = _parse_media("cloth")
            result = run_generate_instantfit_flow(
                GenerateImageInputSchema(
                    system=generate_image_system_prompt,
                    prompt=[
                        Part(
                            root=TextPart(
                                text=generate_image_user_prompt.format(
                                    occasion=occasion,
                                    mood=mood,
                                    time_of_day=time_of_day,
                                )
                            )
                        ),
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url=selfie_url,
                                    content_type=selfie_mimetype,
                                )
                            )
                        ),
                        Part(
                            root=MediaPart(
                                media=Media(
                                    url=cloth_url,
                                    content_type=cloth_mimetype,
                                )
                            )
                        ),
                    ],
                )
            )
        else:
            result = run_generate_autofit_flow(
                GenerateImageInputSchema(
                    system=generate_image_system_prompt,
                    prompt=generate_image_user_prompt.format(
                        occasion=occasion, mood=mood, time_of_day=time_of_day
                    ),
                    uid=uid,
                    firestore_client=db,
                )
            )
    except https_fn.HttpsError:
        raise
    except Exception:
        logger.exception("generate_image: flow execution failed")
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message="Failed to generate image",
        )

    if not result or not result.image_data:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.FAILED_PRECONDITION,
            message="Failed to generate image",
        )

    image_data = result.image_data
    if isinstance(image_data, str):
        try:
            image_data = decode_base64_image(image_data)
        except Exception:
            raise https_fn.HttpsError(
                code=https_fn.FunctionsErrorCode.FAILED_PRECONDITION,
                message="Generated image data is invalid",
            )
    if not isinstance(image_data, bytes):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.FAILED_PRECONDITION,
            message="Generated image data is invalid",
        )
    if not result.content_type.startswith("image/"):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.FAILED_PRECONDITION,
            message="Generated content type is invalid",
        )

    try:
        thumbnail = generate_thumbnail(image_data, (300, 300), blur=False)
        placeholder_image = generate_thumbnail(image_data, (50, 50), blur_radius=3)
        dominant_color = get_dominant_color(image_data)
    except Exception:
        logger.exception("generate_image: post-processing failed")
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message="Failed to process generated image",
        )

    image_blob = None
    thumbnail_blob = None
    filename = pathlib.PurePath(generate_image_filename(result.content_type))
    try:
        image_blob = storage_bucket.blob(f"users/{uid}/outfit/{filename}")
        image_blob.upload_from_string(image_data, content_type=result.content_type)
        image_blob.make_public()
        image_gs_url = f"gs://{image_blob.bucket.name}/{image_blob.name}"
        image_http_url = image_blob.public_url

        thumbnail_blob = storage_bucket.blob(
            f"users/{uid}/outfit_thumbnail/thumb-{filename.stem}.jpeg"
        )
        thumbnail_blob.upload_from_string(
            decode_base64_image(thumbnail), content_type="image/jpeg"
        )
        thumbnail_blob.make_public()
        thumbnail_http_url = thumbnail_blob.public_url

        db.collection(f"users/{uid}/outfit").add(
            {
                "image_url": image_http_url,
                "gs_url": image_gs_url,
                "thumbnail": thumbnail_http_url,
                "placeholder_image": placeholder_image,
                "dominant_color": dominant_color,
                "description": result.description,
                "content_type": result.content_type,
                "created_at": cloud_firestore.SERVER_TIMESTAMP,
            }
        )
    except https_fn.HttpsError:
        raise
    except Exception:
        logger.exception("generate_image: failed while persisting output")
        for blob in (thumbnail_blob, image_blob):
            if blob is None:
                continue
            try:
                blob.delete()
            except Exception:
                logger.exception(
                    "generate_image: failed to clean up blob %s", blob.name
                )
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message="Failed to save generated image",
        )

    return {
        "image": image_http_url,
        "thumbnail": placeholder_image,
        "description": result.description,
    }


@storage_fn.on_object_finalized(timeout_sec=540, memory=MemoryOption.MB_512)  # type: ignore
def generate_embedding(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],
):
    bucket_name = event.data.bucket
    object_name = event.data.name
    if not bucket_name or not object_name:
        logger.warning("generate_embedding: missing bucket or object name")
        return

    file_path = pathlib.PurePath(object_name)
    content_type = event.data.content_type
    source_generation = event.data.generation

    gs_url = f"gs://{bucket_name}/{file_path}"

    # Exit if this is triggered on a file that is not an image.
    if file_path.name.startswith("thumb-"):
        return
    if not content_type or not content_type.startswith("image/"):
        return

    if len(file_path.parts) < 4 or file_path.parts[0] != "users":
        logger.warning(
            "generate_embedding: unexpected object path format: %s", file_path
        )
        return

    uid = file_path.parts[1]
    media_kind = file_path.parts[2]
    if media_kind not in ["clothes", "selfies"]:
        return

    collection_path = f"users/{uid}/{media_kind}"
    db = firestore.client()
    # Best-effort dedupe for retried finalize events.
    query = db.collection(collection_path).where("gs_url", "==", gs_url)
    if source_generation:
        query = query.where("source_generation", "==", source_generation)
    if query.limit(1).get():
        logger.info("generate_embedding: skipping duplicate event for %s", gs_url)
        return

    bucket = storage.bucket(bucket_name)
    image_blob = bucket.blob(str(file_path))
    image_bytes = image_blob.download_as_bytes(raw_download=True)

    placeholder_image = generate_thumbnail(image_bytes, (50, 50), blur_radius=3)

    thumbnail = generate_thumbnail(
        image_bytes, (512, 512), ret_as_bytes=True, blur=False
    )
    thumbnail_path = file_path.parent / pathlib.PurePath(f"thumb-{file_path.stem}.jpeg")
    thumbnail_blob = bucket.blob(str(thumbnail_path))
    thumbnail_blob.upload_from_string(thumbnail, content_type="image/jpeg")

    image_blob.make_public()
    image_url = image_blob.public_url

    thumbnail_blob.make_public()
    thumbnail_url = thumbnail_blob.public_url

    thumbnail_gs_url = f"gs://{thumbnail_blob.bucket.name}/{thumbnail_blob.name}"

    document_data = {
        "image_url": image_url,
        "thumbnail_url": thumbnail_url,
        "gs_url": gs_url,
        "placeholder_image": placeholder_image,
        "created_at": cloud_firestore.SERVER_TIMESTAMP,
        "mimetype": content_type,
        "source_generation": source_generation,
    }

    context = None
    try:
        user = auth.get_user(uid)
        if user.display_name:
            context = user.display_name
    except UserNotFoundError:
        logger.warning("generate_embedding: failed to load auth user for uid=%s", uid)
        return

    if media_kind == "clothes":
        result: ImageCategorizationOutput = run_categorize_image_flow(
            ImageCategorizationInput(
                system=categorization_system_prompt,
                prompt=[
                    Part(root=TextPart(text=categorization_user_prompt)),
                    Part(
                        root=MediaPart(
                            media=Media(
                                url=thumbnail_gs_url,
                                content_type="image/jpeg",
                            )
                        )
                    ),
                ],
            )
        )
        document_data["category"] = result.category
        context = result.category

    run_indexer(
        index_ref="index_documents",
        documents=[
            Document(
                content=[
                    DocumentPart(
                        root=MediaPart(
                            media=Media(
                                url=gs_url,
                                content_type=content_type,
                            ),
                        )
                    ),
                    DocumentPart(root=TextPart(text=prepare_document(context))),
                ]
            ),
        ],
        options={
            "": IndexData(
                vector_field="embedding",
                collection=collection_path,
                document_data=document_data,
                options={"output_dimensionality": 768},
            )
        },
    )

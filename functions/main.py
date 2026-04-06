# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
import pathlib

from firebase_admin import initialize_app, storage, firestore, auth
from firebase_functions import https_fn, storage_fn
from firebase_functions.options import set_global_options, MemoryOption
from genkit import Part, TextPart, MediaPart, Media, Document
from genkit.core.typing import DocumentPart
from google.cloud import firestore as cloud_firestore

from models import (
    GenerateImageInputSchema,
    ImageConfig,
    ImageGenerationResult,
    ImageCategorizationInput,
    ImageCategorizationOutput,
    IndexData,
)
from prompts import (
    instantfit_system_prompt2,
    categorization_system_prompt,
    categorization_user_prompt,
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
)

# For cost control, you can set the maximum number of containers that can be
# running at the same time. This helps mitigate the impact of unexpected
# traffic spikes by instead downgrading performance. This limit is a per-function
# limit. You can override the limit for each function using the max_instances
# parameter in the decorator, e.g. @https_fn.on_request(max_instances=5).
set_global_options(max_instances=10)
initialize_app()


@https_fn.on_call(timeout_sec=600, memory=MemoryOption.MB_512, cpu=2)
def generate_image(req: https_fn.CallableRequest) -> dict:
    if not req.auth.uid:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.UNAUTHENTICATED,
            message="You're not allowed here",
        )
    if not req.data.get("prompt"):
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Prompt is required",
        )
    db = firestore.client()
    result: ImageGenerationResult | None
    if req.data.get("type") == "instantfit":
        result = run_generate_instantfit_flow(
            GenerateImageInputSchema(
                system=instantfit_system_prompt2,
                prompt=[
                    Part(root=TextPart(text=req.data.get("prompt"))),
                    Part(
                        root=MediaPart(
                            media=Media(
                                url=req.data.get("selfie").get("gs_url"),
                                content_type=req.data.get("selfie").get("mimetype"),
                            )
                        )
                    ),
                    Part(
                        root=MediaPart(
                            media=Media(
                                url=req.data.get("cloth").get("gs_url"),
                                content_type=req.data.get("cloth").get("mimetype"),
                            )
                        )
                    ),
                ],
                image_config=ImageConfig(),
            )
        )
    else:
        result = run_generate_autofit_flow(
            GenerateImageInputSchema(
                system=instantfit_system_prompt2,
                prompt=req.data.get("prompt"),
                uid=req.auth.uid,
                image_config=ImageConfig(),
                firestore_client=db,
            )
        )

    if not result or not result.image_data:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.FAILED_PRECONDITION,
            message="Failed to generate image",
        )

    thumbnail = generate_thumbnail(result.image_data, (300, 300), blur=False)
    placeholder_image = generate_thumbnail(result.image_data, (50, 50), blur_radius=3)
    dominant_color = get_dominant_color(result.image_data)

    storage_bucket = storage.bucket()
    filename = pathlib.PurePath(generate_image_filename(result.content_type))
    image_blob = storage_bucket.blob(f"users/{req.auth.uid}/outfit/{filename}")
    image_blob.upload_from_string(result.image_data, content_type=result.content_type)
    image_blob.make_public()
    image_gs_url = f"gs://{image_blob.bucket.name}/{image_blob.name}"
    image_http_url = image_blob.public_url

    thumbnail_blob = storage_bucket.blob(
        f"users/{req.auth.uid}/outfit_thumbnail/thumb-{filename.stem}.jpeg"
    )
    thumbnail_blob.upload_from_string(
        decode_base64_image(thumbnail), content_type="image/jpeg"
    )
    thumbnail_blob.make_public()
    thumbnail_http_url = thumbnail_blob.public_url

    db.collection(f"users/{req.auth.uid}/outfit").add(
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

    return {
        "image": image_http_url,
        "thumbnail": placeholder_image,
        "description": result.description,
    }


@storage_fn.on_object_finalized(timeout_sec=540, memory=MemoryOption.MB_512)
def generate_embedding(
    event: storage_fn.CloudEvent[storage_fn.StorageObjectData],
):
    bucket_name = event.data.bucket
    file_path = pathlib.PurePath(event.data.name)
    content_type = event.data.content_type

    gs_url = f"gs://{bucket_name}/{file_path}"

    # Exit if this is triggered on a file that is not an image.
    if file_path.name.startswith("thumb-"):
        return
    if not content_type or not content_type.startswith("image/"):
        return

    if file_path.parts[2] not in ["clothes", "selfies"]:
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
    }

    user = auth.get_user(file_path.parts[1])
    context = user.display_name

    if file_path.parts[2] == "clothes":
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
                                content_type="image/jpeg",
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
                collection=f"users/{file_path.parts[1]}/{file_path.parts[2]}",
                document_data=document_data,
                options={"output_dimensionality": 768},
            )
        },
    )

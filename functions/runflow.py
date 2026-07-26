from genkit import GenkitError
from google.genai.errors import ClientError
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
)

from ai import flow_ai
from flows import generate_instantfit_flow, categorize_image_flow, generate_autofit_flow
from models import (
    ImageGenerationResult,
    GenerateImageInputSchema,
    ImageCategorizationInput,
    ImageCategorizationOutput,
)


def run_generate_instantfit_flow(
    input_schema: GenerateImageInputSchema,
) -> ImageGenerationResult | None:
    return flow_ai.run_main(generate_instantfit_flow(input_schema))


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((GenkitError, ClientError)),
)
def run_categorize_image_flow(
    input_schema: ImageCategorizationInput,
) -> ImageCategorizationOutput:
    return flow_ai.run_main(categorize_image_flow(input_schema))


def run_generate_autofit_flow(
    input_schema: GenerateImageInputSchema,
) -> ImageGenerationResult | None:
    return flow_ai.run_main(generate_autofit_flow(input_schema))


__all__ = [
    "run_generate_instantfit_flow",
    "run_categorize_image_flow",
    "run_generate_autofit_flow",
]


if __name__ == "__main__":
    from prompts import (
        generate_image_system_prompt,
    )
    from firebase_admin import initialize_app, firestore
    from models import ImageConfig

    initialize_app()

    selfie = "gs://my-aurafit.firebasestorage.app/users/INOv1CMZ5aXRh2Q5OF6ZRB7XMAB2/selfie/1000411251.jpg"
    cloth = "gs://my-aurafit.firebasestorage.app/users/INOv1CMZ5aXRh2Q5OF6ZRB7XMAB2/cloth/1000428418.jpg"

    # print(
    #     run_generate_instantfit_flow(
    #         GenerateImageInputSchema(
    #             system=instantfit_system_prompt2,
    #             prompt=[
    #                 Part(root=TextPart(text="movies")),
    #                 Part(
    #                     root=MediaPart(
    #                         media=Media(url=selfie, content_type="image/jpeg")
    #                     )
    #                 ),
    #                 Part(
    #                     root=MediaPart(
    #                         media=Media(url=cloth, content_type="image/jpeg")
    #                     )
    #                 ),
    #             ],
    #             image_config=ImageConfig(),
    #         )
    #     )
    # )

    # print(
    #     run_categorize_image_flow(
    #         ImageCategorizationInput(
    #             system=categorization_system_prompt,
    #             prompt=[
    #                 Part(root=TextPart(text=categorization_user_prompt)),
    #                 Part(
    #                     root=MediaPart(
    #                         media=Media(
    #                             url="gs://my-aurafit.firebasestorage.app/users/tZPDMCAB1nUKDBlAkO1MazhFeeI3/cloth/1000374291.jpg",
    #                             content_type="image/jpeg",
    #                         )
    #                     )
    #                 ),
    #             ],
    #         )
    #     )
    # )


    result = run_generate_autofit_flow(
        GenerateImageInputSchema(
            uid="NlGVVbZ1ssMaOROGB9giY0Yu4KB3",
            system=generate_image_system_prompt,
            prompt="I'm going for an outdoor event and it's kind of cold",
            image_config=ImageConfig(),
            firestore_client=firestore.client(),
        )
    )

    with open("test.png", "wb") as f:
        f.write(result.image_data)

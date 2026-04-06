from genkit import Part, Output, MediaPart, Media, TextPart

from ai import flow_ai, indexer_ai
from models import (
    GenerateImageInputSchema,
    ImageGenerationResult,
    ImageCategorizationOutput,
    ImageCategorizationInput,
    QueryOutput,
    ImageConfig,
    UserDataSearchInput,
)
from prompts import user_cloth_vector_search_system_prompt
from retriever import user_clothes_retriever, user_selfie_retriever
from utils import (
    decode_base64_image,
)


@flow_ai.flow()
async def categorize_image_flow(
    input_schema: ImageCategorizationInput,
) -> ImageCategorizationOutput:
    """Flow to categorize an image into predefined categories.

    Args:
        input_schema (ImageCategorizationInput): Input schema for image categorization.

    Returns:
        ImageCategorizationOutput: Output schema with categorized image.
    """
    response = await flow_ai.generate(
        system=input_schema.system,
        model=input_schema.model,
        prompt=input_schema.prompt,
        output=Output(schema=ImageCategorizationOutput),
        config={
            "response_modalities": ["TEXT"],
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        },
    )

    return response.output


@flow_ai.flow()
async def generate_autofit_flow(
    input_schema: GenerateImageInputSchema,
) -> ImageGenerationResult:
    async def generate_query(user_prompt) -> str:
        result = await flow_ai.generate(
            system=user_cloth_vector_search_system_prompt,
            model="vertexai/gemini-3-flash-preview",
            prompt=user_prompt,
            output=Output(schema=QueryOutput),
            config={
                "response_modalities": ["TEXT"],
                "safety_settings": [
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE",
                    },
                ],
            },
        )
        return result.output.query

    async def user_data_vector_search(
        query_input: UserDataSearchInput,
    ) -> list[dict[str, str]]:
        retrieved = await indexer_ai.retrieve(
            retriever=query_input.retriever,
            query=query_input.query,
            options=query_input.options,
        )
        return [
            {"gs_url": doc.metadata["gs_url"], "mimetype": doc.metadata["mimetype"]}
            for doc in retrieved.documents
        ]

    query = await flow_ai.run("generate-query", input_schema.prompt, generate_query)
    clothes = await flow_ai.run(
        "user-cloth-vector-search",
        UserDataSearchInput(
            retriever=user_clothes_retriever(
                input_schema.uid, input_schema.firestore_client
            ),
            query=query,
            options={"limit": 20},
        ),
        user_data_vector_search,
    )
    selfie = await flow_ai.run(
        "user-selfie-vector-search",
        UserDataSearchInput(
            retriever=user_selfie_retriever(
                input_schema.uid, input_schema.firestore_client
            ),
            query="A clear, high-resolution front-facing portrait of a person. "
            "Close-up shot with a focus on the face, well-lit with natural lighting, "
            "sharp features, and a clean background. Authentic facial expression, "
            "high-quality mobile photography, centered composition, and a shallow depth of field. "
            "A centered selfie with no obstructions, professional-grade skin tones, and vibrant clarity.",
            options={"limit": 5},
        ),
        user_data_vector_search,
    )
    prompt = [
        Part(
            root=MediaPart(
                media=Media(
                    url=doc["gs_url"],
                    content_type=doc["mimetype"],
                ),
            )
        )
        for doc in selfie
    ]
    prompt += [
        Part(
            root=MediaPart(
                media=Media(
                    url=doc["gs_url"],
                    content_type=doc["mimetype"],
                ),
            )
        )
        for doc in clothes
    ]
    prompt.append(Part(root=TextPart(text=input_schema.prompt)))
    input_schema.prompt = prompt
    return await generate_instantfit_flow(input_schema)


@flow_ai.flow()
async def generate_instantfit_flow(
    input_schema: GenerateImageInputSchema,
) -> ImageGenerationResult | None:
    response = await flow_ai.generate(
        system=input_schema.system,
        model=input_schema.model,
        prompt=input_schema.prompt,
        config={
            "response_modalities": ["IMAGE", "TEXT"],
            "image_config": input_schema.image_config.model_dump(),
            "safety_settings": [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
            ],
        },
    )

    if response.message and response.message.content:
        contents = response.message.content
        result = ImageGenerationResult(image_data=b"", description="")
        for content in contents:
            if text := content.root.text:
                result.description += text
            if media := content.root.media:
                image_data = decode_base64_image(media.url)
                result.image_data = image_data
                result.content_type = media.content_type
                # with open("outfit1.png", "wb") as f:
                #     f.write(image_data)
        return result
    return None


if __name__ == "__main__":
    from firebase_admin import firestore, initialize_app
    from prompts import instantfit_system_prompt2

    initialize_app()
    selfie_url = "gs://my-aurafit.firebasestorage.app/users/INOv1CMZ5aXRh2Q5OF6ZRB7XMAB2/selfie/1000411251.jpg"
    cloth_url = "gs://my-aurafit.firebasestorage.app/users/INOv1CMZ5aXRh2Q5OF6ZRB7XMAB2/cloth/1000428418.jpg"

    # ai.run_main(
    #     generate_instantfit_flow(
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
    #     flow_ai.run_main(
    #         categorize_image_flow(
    #             ImageCategorizationInput(
    #                 system=categorization_system_prompt,
    #                 prompt=[
    #                     Part(root=TextPart(text=categorization_user_prompt)),
    #                     Part(
    #                         root=MediaPart(
    #                             media=Media(
    #                                 url="gs://my-aurafit.firebasestorage.app/users/tZPDMCAB1nUKDBlAkO1MazhFeeI3/cloth/1000374291.jpg",
    #                                 content_type="image/jpeg",
    #                             )
    #                         )
    #                     ),
    #                 ],
    #             )
    #         )
    #     )
    # )

    print(
        flow_ai.run_main(
            generate_autofit_flow(
                GenerateImageInputSchema(
                    uid="NlGVVbZ1ssMaOROGB9giY0Yu4KB3",
                    system=instantfit_system_prompt2,
                    prompt="I'm going for an outdoor event and it's kind of cold",
                    image_config=ImageConfig(),
                    firestore_client=firestore.client(),
                )
            )
        )
    )

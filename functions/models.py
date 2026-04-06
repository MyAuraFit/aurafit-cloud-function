from typing import Literal, Any, Annotated

from genkit import Part
from google.cloud import firestore
from pydantic import BaseModel, Field, ConfigDict
from pydantic.json_schema import SkipJsonSchema


class ImageConfig(BaseModel):
    aspect_ratio: str = Field(
        default="3:4", description="The aspect ratio of the image."
    )
    image_size: str | None = Field(default=None, description="The size of the image.")


class GenerateImageInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    uid: str | None = Field(
        default=None, description="The user ID to search clothes for."
    )
    system: str = Field(
        description="The system prompt for the image generation.",
    )
    prompt: str | Part | list[Part] | None = Field(
        default=None, description="The prompt to generate an image for."
    )
    model: str = Field(
        default="vertexai/gemini-2.5-flash-image",
        description="The model to use for image generation.",
    )
    image_config: ImageConfig = Field(
        description="The configuration for the image generation."
    )
    firestore_client: Annotated[firestore.Client | None, SkipJsonSchema()] = Field(
        default=None, description="The firestore client to use.", exclude=True
    )


class ImageCategorizationOutput(BaseModel):
    category: Literal[
        "Casual",
        "Formal",
        "Business",
        "Traditional",
        "Streetwear",
        "Sportswear",
        "Activewear",
        "Loungewear",
        "Sleepwear",
        "Underwear",
        "Outerwear",
        "Workwear",
        "Uniform wear",
        "Resort wear",
        "Beachwear",
        "Swimwear",
        "Party wear",
        "Evening wear",
        "Bridal wear",
        "Maternity wear",
        "Plus size",
        "Petite wear",
        "Tall wear",
        "Vintage",
        "Luxury wear",
        "Smart casual",
        "Business casual",
        "Semi formal",
        "Festival wear",
        "Costume wear",
        "Modest wear",
        "Religious wear",
        "Adaptive wear",
        "Performance wear",
        "Protective wear",
        "Travel wear",
        "Winter wear",
        "Summer wear",
        "Rain wear",
    ] = Field(description="The category of the image.")


class ImageCategorizationInput(BaseModel):
    system: str = Field(description="The system prompt for image categorization.")
    prompt: str | Part | list[Part] = Field(
        description="The prompt image to categorize."
    )
    model: str = Field(
        default="vertexai/gemini-3-flash-preview",
        description="The model to use for image categorization.",
    )


class ImageGenerationResult(BaseModel):
    image_data: bytes | str = Field(
        description="The image data to upload.",
        exclude=True,
    )
    content_type: str = Field(
        default="image/png", description="The content type of the image."
    )
    description: str = Field(description="A description of the generated image.")


class IndexData(BaseModel):
    embedder: str | None = Field(
        default="vertexai/gemini-embedding-2-preview",
        description="The embedder model to use for generating the embedding.",
    )
    vector_field: str = Field(
        description="The name of the vector field in the vector store."
    )
    metadata: dict[str, object] | None = Field(
        default=None, description="Additional metadata for the document."
    )
    options: dict[str, object] | None = Field(
        default=None, description="task_type for the embedding request."
    )
    collection: str | None = Field(
        default=None, description="The collection to store the embedded document in."
    )
    document_data: dict[str, Any] | None = Field(
        description="The original document data to store in the firestore.",
        exclude=True,
    )


class UserDataSearchInput(BaseModel):
    retriever: str = Field(description="The retriever name to use.")
    query: str | None = Field(description="The search query to find relevant clothes.")
    options: dict[str, object] | None = Field(description="retriever options.")


class QueryOutput(BaseModel):
    query: str = Field(description="Generated search query.")

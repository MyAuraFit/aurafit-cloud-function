from genkit import Document, MediaPart, Media, TextPart, GenkitError
from genkit.core.typing import DocumentPart
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
)

from indexer import indexer_ai
from models import IndexData


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((RuntimeError, GenkitError)),
)
def run_indexer(
    index_ref: str, documents: list[Document], options: dict[str, IndexData]
):
    indexer_ai.run_main(
        indexer_ai.index(indexer=index_ref, documents=documents, options=options)
    )


if __name__ == "__main__":
    docs = [
        Document(
            content=[
                DocumentPart(
                    root=MediaPart(
                        media=Media(
                            url="gs://my-aurafit.firebasestorage.app/users/INOv1CMZ5aXRh2Q5OF6ZRB7XMAB2/cloth/1000428418.jpg",
                            content_type="image/jpeg",
                        ),
                    )
                ),
                DocumentPart(
                    root=TextPart(
                        text="This is an image of a cloth that can be used as an ingredient in the recipe."
                    )
                ),
            ]
        ),
    ]

    run_indexer(
        index_ref="index_documents",
        documents=docs,
        options={
            "": IndexData(
                vector_field="",
                metadata=None,
                # options={"task_type": "RETRIEVAL_DOCUMENT"},
                collection="run_indexer",
                document_data={},
            )
        },
    )

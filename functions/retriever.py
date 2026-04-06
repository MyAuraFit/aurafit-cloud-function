from genkit.plugins.firebase import define_firestore_vector_store

from ai import indexer_ai


# initialize_app()
# firestore_client = firestore.client()


def user_clothes_retriever(uid, firestore_client):
    return define_firestore_vector_store(
        ai=indexer_ai,
        name="clothes_retriever",
        embedder="vertexai/gemini-embedding-2-preview",
        collection=f"users/{uid}/clothes",
        vector_field="embedding",
        content_field="category",
        firestore_client=firestore_client,
        embedder_options={"output_dimensionality": 768},
    )


def user_selfie_retriever(uid, firestore_client):
    return define_firestore_vector_store(
        ai=indexer_ai,
        name="selfie_retriever",
        embedder="vertexai/gemini-embedding-2-preview",
        collection=f"users/{uid}/selfies",
        vector_field="embedding",
        content_field="created_at",
        firestore_client=firestore_client,
        embedder_options={"output_dimensionality": 768},
    )

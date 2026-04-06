from firebase_admin import firestore_async
from genkit import ActionRunContext
from genkit.blocks.retriever import IndexerRequest
from google.cloud.firestore_v1.vector import Vector

from ai import indexer_ai


async def index_documents(input_req: IndexerRequest, ctx: ActionRunContext):
    documents = input_req.documents
    options = input_req.options
    for doc, index_data in zip(documents, options.values()):
        embedding = await indexer_ai.embed(
            embedder=index_data.embedder,
            metadata=index_data.metadata,
            options=index_data.options,
            content=doc,
        )
        if not (
            index_data.document_data
            and index_data.vector_field
            and index_data.collection
        ):
            return
        index_data.document_data[index_data.vector_field] = Vector(
            embedding[0].embedding
        )
        db = firestore_async.client()
        await db.collection(index_data.collection).add(index_data.document_data)


indexer_ai.define_indexer(name="index_documents", fn=index_documents)  # type: ignore

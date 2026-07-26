from genkit import Genkit
from genkit.plugins.google_genai import VertexAI

flow_ai = Genkit(
    plugins=[VertexAI(project="my-aurafit", location="global")],
)
indexer_ai = Genkit(
    plugins=[VertexAI(project="my-aurafit", location="global")],
)

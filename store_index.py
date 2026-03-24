from dotenv import load_dotenv
import os
load_dotenv()
from src.helper import load_all_data,filter_to_minimal_docs,download_embeddings,text_split
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

faq_data=[ {"q": "What is arrhythmia?", "a": "Irregular heartbeat condition."},
    {"q": "What are symptoms?", "a": "Dizziness, chest pain, palpitations."},
    {"q": "What causes arrhythmia?", "a": "Heart disease, stress, BP."}]
documents = load_all_data("data", faq_data)
minimal_docs=filter_to_minimal_docs(documents)
text_chunk=text_split(minimal_docs)

embedding=download_embeddings()
pinecode_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecode_api_key)
index_name="chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384, #dimension of embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud='aws',region="us-east-1")
    )
index=pc.Index(index_name)

docsearch=PineconeVectorStore.from_documents(
    documents=text_chunk,
    embedding=embedding,
    index_name=index_name
)
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

#extract data from files
def load_all_data(data,faq_data=None):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    if not documents:
        print(" No PDF files found in the directory.")
    else:
        print(f"Loaded {len(documents)} documents")

    faq_documents= []
    if faq_data:
        for item in faq_data:
            text = f"Question: {item['q']}\nAnswer: {item['a']}"
            faq_documents.append(
                Document(
                    page_content=text,
                    metadata={"source": "FAQ"}
                )
            )

        print(f" Loaded {len(faq_documents)} FAQ entries")


    all_documents =documents + faq_documents

    print(f"Total documents: {len(all_documents)}")

    return all_documents

#filter docs
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(page_content=doc.page_content,
                     metadata={"source": src})
        )
    return minimal_docs

# split docs into chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunk=text_splitter.split_documents(minimal_docs)
    return text_chunk

#download embedding model
from langchain.embeddings import HuggingFaceEmbeddings
def download_embeddings():
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

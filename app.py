from flask import Flask ,render_template,jsonify,request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os


app=Flask(__name__)

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

embeddings=download_embeddings()
index_name="chatbot"
docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":4})
llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
    max_tokens=150
)
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}")
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retriever,question_answer_chain)

@app.route("/")
def index():
    return render_template("bot.html")

@app.route("/chat",methods=["GET","POST"])
def chat():
    data = request.get_json()
    question = data["question"]

    response = rag_chain.invoke({"input": question})
    answer = response.get("answer", "")

    if not answer.strip():
        answer = "I don't have enough information."

    return jsonify({"answer": answer})
if __name__=='__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)


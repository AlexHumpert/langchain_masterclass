# File will be used to allow us to run queries against the uni_embs directory
# "uni_embs" directory contains SQLite vector database with vector embeddings of "global_university_rankings.csv" document

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Creating chat model instance
chat = ChatOpenAI()

#Creating embeddings instance
embeddings = OpenAIEmbeddings()

# Creating instance of chroma to enable query to refer to embedded .csv
db = Chroma(
    persist_directory = "uni_embs",
    embedding_function = embeddings
)

# Creating retriver instance to connect db to chain
retriever = db.as_retriever()

# Creating chain instance by passing chat model, retriver and chain_type
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
)

while True:
    content = input("> ")
    result = chain.run(content)
    print(result)
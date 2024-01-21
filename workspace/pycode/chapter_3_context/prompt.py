# File you run any time you want to ask a question to GPT and use content inside vector db for context
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Creating chat model instance
chat = ChatOpenAI()

# Creating embeddings instance
embeddings = OpenAIEmbeddings()

# We don't want to recreate embeddings
# We just want instance of chroma to allow us to do similarity search
db = Chroma(
    persist_directory = "emb",
    embedding_function = embeddings
)

# Creating retriever instance
retriever = db.as_retriever()

# Creating chain instance by passing chat model, retriever and chain type
chain = RetrievalQA.from_chain_type(
    llm = chat,
    retriever = retriever,
    chain_type = "stuff"
)

#result = chain.run("what is an interesting fact about the English language")

#print(result)

while True:
    content = input("> ")
    result = chain.run(content)
    print(result)
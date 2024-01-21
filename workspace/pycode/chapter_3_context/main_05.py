from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0
)

loader = TextLoader("facts.txt")

docs = loader.load_and_split(
    text_splitter = text_splitter
)


db = Chroma.from_documents(
    docs, # creates chroma instance and calculates embeddings for all docs
    embedding = embeddings,
    persist_directory = "emb" # persists embeddings in SQLite directory on computer
)

results = db.similarity_search( #Prompt is called to the db instance
    "What is an interesting fact about the English language",
    #k = 1 # Number of results you get back
    )



for result in results: # results returns an array of tuples (if you use similarity_search_with_score)
    print("\n")
    print(result.page_content)
    #print(result[1])
    #print(result[0].page_content)

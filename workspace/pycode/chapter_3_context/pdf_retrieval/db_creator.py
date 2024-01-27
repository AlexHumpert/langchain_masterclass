# I want to create a local SQLite vector database storing embeddings of Julie's first Oxford paper
# This script "main_06.py" will create the SQLite database and save it locally in a folder named "uni_embs"
# My next script "prompt_01.py" will use a langchain retriever to connect a chain to the database so user
# queries can be matched against it.

from langchain_community.document_loaders.pdf import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Creating an embeddings instance with OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Creating the textplitter and settig some basic parameters
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0
)


# Loading the pdf to the "loader" variable
#loader = PyPDFLoader(
#    file_path = "AIES3039_Extractivist_and_exploitative_labour_practices.pdf"
#)

# Loading directory with "pdf" files
loader = PyPDFDirectoryLoader(
    "pdfs/"
)

# Creating the documents we will embed
# Docs are chunks of the csv file chunked as per the text_splitter parameters
docs = loader.load_and_split(
    text_splitter = text_splitter
)

# create chroma db instance 
db = Chroma.from_documents(
    docs,
    embedding = embeddings,
    persist_directory = "pdf_embs"
)
from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

while True:
    content = input("> ")
    query_result = embeddings.embed_query(content)
    print(query_result)
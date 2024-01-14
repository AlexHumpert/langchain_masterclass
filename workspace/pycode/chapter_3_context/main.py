from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("facts.txt")

splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0
)

docs = loader.load_and_split(
    text_splitter = splitter
)

for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
    print("\n")

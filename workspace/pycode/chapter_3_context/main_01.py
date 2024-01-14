from langchain.document_loaders import TextLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader("global_university_ranks.csv")

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

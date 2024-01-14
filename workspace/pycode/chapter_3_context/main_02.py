from langchain.document_loaders import TextLoader, CSVLoader, GitHubIssuesLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

loader = GitHubIssuesLoader(
    repo = "AlexHumpert/langchain_masterclass",
    creator = "AlexHumpert"

)

splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0
)

docs = loader.load()

print(docs)
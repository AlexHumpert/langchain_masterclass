from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from

from dotenv import load_dotenv

load_dotenv()

loader = GitHubIssuesLoader(
    repo = "AlexHumpert/langchain_masterclass",
    creator = "AlexHumpert",
    state = "all"
)

splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 200,
    chunk_overlap = 0
)

docs = loader.load()

print(docs[0].page_content)



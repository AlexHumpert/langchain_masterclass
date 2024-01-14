from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("hello there my name is Alex")

print(embed)
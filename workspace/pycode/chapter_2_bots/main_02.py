from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory, ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(
    verbose = True
)

memory = ConversationSummaryMemory(
    # chat_memory = FileChatMessageHistory("messages.json"),
    memory_key = "messages", 
    return_messages = True,
    llm = chat, )

# This is what make the "nex word prediction" LLM have a conversational tone
# This is similar to a prompt template in the sequential chains
prompt = ChatPromptTemplate(
    input_variables = ["content", "messages"],
    messages = [
        MessagesPlaceholder(variable_name = "messages"), 
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    memory = memory,
    verbose = True
)

while True:
    content = input("> ")
    result = chain({"content" : content})
    print(result["text"])



    
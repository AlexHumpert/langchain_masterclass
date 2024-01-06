from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# SECURE THIS KEY!
api_key = "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"

llm = OpenAI(
    openai_api_key=api_key
)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)

result = code_chain({
    "language": "python",
    "task": "return a list of numbers"
})

print(result["text"])

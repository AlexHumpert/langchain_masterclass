from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default = "return a list of numbers")
parser.add_argument("--language", default = "python")
args = parser.parse_args()


# Secure this key
api_key = "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"

llm = OpenAI(openai_api_key=api_key)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(
    llm=llm, 
    prompt=code_prompt
    )

result = code_chain({
    "language": args.language,
    "task": args.task
    })

print(result["text"])

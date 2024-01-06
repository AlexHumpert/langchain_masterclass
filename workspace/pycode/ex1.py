from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", default = "return a list of numbers")
parser.add_argument("--language", default = "python")
parser.add_argument("--language_2", default = "english")
args = parser.parse_args()


# Secure this key
api_key = "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"

llm = OpenAI(openai_api_key=api_key)

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}. Explain the {task} in {language_2}",
    input_variables=["language", "task", "language_2"],
)

code_chain = LLMChain(
    llm=llm, 
    prompt=code_prompt
    )

result = code_chain({
    "language" : args.language,
    "task" : args.task,
    "language_2" : args.language_2
    })

print(result["text"])

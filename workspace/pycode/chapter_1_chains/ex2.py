from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--concept", default = "machine learning")
parser.add_argument("--audience_1", default = "five-year old")
parser.add_argument("--audience_2", default = "expert in the field")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Explain this {concept} in tailored to each of the following audience members: {audience_1}, {audience_2}",
    input_variables=["concept", "audience_1", "audience_2"],
)

code_chain = LLMChain(
    llm=llm, 
    prompt=code_prompt
    )

result = code_chain({
    "concept": args.concept,
    "audience_1": args.audience_1,
    "audience_2" : args.audience_2
    })

print(result["text"])

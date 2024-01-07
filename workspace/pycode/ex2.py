from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--person_1", default = "Benjamin Franklin")
parser.add_argument("--person_2", default = "Thomas Jefferson")
parser.add_argument("--topic", default = "democracy")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short script of an argument between {person_1} and {person_2} about {topic}.",
    input_variables=["person_1", "person_2", "topic"],
)

code_chain = LLMChain(
    llm=llm, 
    prompt=code_prompt
    )

result = code_chain({
    "person_1": args.person_1,
    "person_2": args.person_2,
    "topic" : args.topic
    })

print(result["text"])

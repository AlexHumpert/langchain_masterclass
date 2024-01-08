from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--person_1", default = "Benjamin Franklin")
parser.add_argument("--person_2", default = "Thomas Jefferson")
parser.add_argument("--person_3", default = "Noam Chomsky")
parser.add_argument("--topic", default = "democracy")
args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short script of an argument between {person_1} and {person_2} about {topic}.",
    input_variables=["person_1", "person_2", "topic"],
)

reaction_prompt = PromptTemplate(
    template = "How would {person_3} react to the {script}.",
    input_variables = ["script"]
)

code_chain = LLMChain(
    llm=llm, 
    prompt=code_prompt,
    output_key = "script"
)

reaction_chain = LLMChain(
    llm = llm,
    prompt = reaction_prompt,
    output_key = "reaction"
)

chain = SequentialChain(
    chains = [code_chain, reaction_chain],
    input_variables = ["person_1", "person_2", "person_3", "topic"],
    output_variables = ["script", "reaction"]
)


result = chain({
    "person_1": args.person_1,
    "person_2": args.person_2,
    "person_3": args.person_3,
    "topic" : args.topic
    })


print(">>>> PERSON 1")
print(result["person_1"])

print(">>>> PERSON 2")
print(result["person_2"])

print(">>>> PERSON 3")
print(result["person_3"])

print(">>>> TOPIC")
print(result["topic"])

print(">>>> SCRIPT")
print(result["script"])

print(">>>> REACTION")
print(result["reaction"])
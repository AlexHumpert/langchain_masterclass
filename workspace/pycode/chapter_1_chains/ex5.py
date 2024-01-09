from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--subject", default = "technology entrepreneurship")
parser.add_argument("--method", default = "DiSSS")
parser.add_argument("--learning_hours", default = "2")
parser.add_argument("--expertise_level", default = "Expert")

args = parser.parse_args()

llm = OpenAI()

learning_goal_method_prompt = PromptTemplate(
    template="Describe how to go about learning {subject} using the {method} learning method.",
    input_variables=["subject", "method"]
)

parameter_prompt = PromptTemplate(
    template="Based on available learning hours per day: {learning_hours} and desired level of expertise: {expertise_level} how would you implement {learning_approach}.",
    input_variables=["description"]
)


learning_goal_method_chain = LLMChain(
    llm=llm, 
    prompt=learning_goal_method_prompt,
    output_key = "learning_approach"
)

parameter_chain = LLMChain(
    llm=llm, 
    prompt=parameter_prompt,
    output_key = "implementation"
)



chain = SequentialChain(
    chains = [learning_goal_method_chain, parameter_chain],
    input_variables = ["subject", "method", "learning_hours", "expertise_level"],
    output_variables = ["learning_approach", "implementation"]
)


result = chain({
    "subject": args.subject,
    "method": args.method,
    "learning_hours": args.learning_hours,
    "expertise_level": args.expertise_level
    })


print(">>>>> SUBJECT")
print(result["subject"])

print(">>>>> METHOD")
print(result["method"])

print(">>>>> LEARNING HOURS")
print(result["learning_hours"])

print(">>>>> EXPERTISE LEVEL")
print(result["expertise_level"])

print(">>>>> IMPLEMENTATION ")
print(result["learning_approach"])

print(">>>>> IMPLEMENTATION ")
print(result["implementation"])



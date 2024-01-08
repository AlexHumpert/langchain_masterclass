from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--success_type", default = "financial")
parser.add_argument("--century", default = "21st")
parser.add_argument("--number", default = 4)
parser.add_argument("--audience", default = "middle aged retired mothers with lots of spare time")

args = parser.parse_args()

llm = OpenAI()

advantage_prompt = PromptTemplate(
    template="Write a short descripion of the the advantages of becoming a {success_type} in the {century} century?",
    input_variables=["success_type", "century"]
)

how_to_prompt= PromptTemplate(
    template = "Write a practical how-to guide to achieving this {success_type} based on the {description}",
    input_variables = ["description"]
)

variation_prompt = PromptTemplate(
    template = "Split up the {how_to} guide into {number} catchy one-liners tailored to the {audience} audience on instagram.",
    input_variables = ["how_to"]
)

advantage_chain = LLMChain(
    llm=llm, 
    prompt=advantage_prompt,
    output_key = "description"
)

how_to_chain = LLMChain(
    llm = llm,
    prompt = how_to_prompt,
    output_key = "how_to"
)

variation_chain = LLMChain(
    llm = llm,
    prompt = variation_prompt,
    output_key = "one_liners"
)


chain = SequentialChain(
    chains = [advantage_chain, how_to_chain, variation_chain],
    input_variables = ["success_type", "century", "number", "audience"],
    output_variables = ["description", "how_to", "one_liners"],
)


result = chain({
    "success_type": args.success_type,
    "century": args.century,
    "number": args.number,
    "audience": args.audience,  

    })


print(">>>>> SUCCESS TYPE")
print(result["success_type"])

print(">>>>> CENTURY")
print(result["century"])

print(">>>>> NUMBER")
print(result["number"])

print(">>>>> AUDIENCE")
print(result["audience"])

print(">>>>> DESCRIPTION")
print(result["description"])

print(">>>>> HOW-TO")
print(result["how_to"])

print(">>>>> ONE-LINERS")
print(result["one_liners"])


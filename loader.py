from dotenv import load_dotenv
import getpass
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    token = None
elif not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

#################################################################################################################

import json
from collections import defaultdict

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def infer_schema(records):
    schema = defaultdict(set)

    for record in records:
        for key, value in record.items():
            schema[key].add(type(value).__name__)

    return {
        field: list(types)
        for field, types in schema.items()
    }

#################################################################################################################

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    task="text-generation",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=512,
    temperature=0.2,
    do_sample=False,
    provider="auto",
    #model_kwargs={"quantization_config": quantization_config},
)
#from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(
#     task="text-generation",
#     model_id="mistralai/Mistral-7B-Instruct-v0.2",
#     max_new_tokens=512,
#     temperature=0.2,
#     do_sample=False,
#     provider="auto",
#     #model_kwargs={"quantization_config": quantization_config},
# )

chat_model = ChatHuggingFace(llm=llm)

#################################################################################################################

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["schema", "sample"],
    template="""
You are a data architect agent.

Given the following inferred JSON schema and sample data:

SCHEMA:
{schema}

SAMPLE:
{sample}

Your tasks:
1. Describe the dataset and its purpose.
2. Provide insights about data quality, cardinality, and possible use cases.
3. Propose a normalized SQL schema.
4. Output valid SQL CREATE TABLE statements.

Use PostgreSQL-compatible SQL.
"""
)

#################################################################################################################

from langchain_core.tools import Tool

json_tool = Tool(
    name="JSONSchemaAnalyzer",
    func=infer_schema,
    description=(
        "Infers the schema of a JSON dataset. "
        "Input must be a list of JSON objects. "
        "Output is a mapping of field names to observed data types."
    )
)

#################################################################################################################

def run_agent(json_path):
    records = load_json(json_path)
    inferred_schema = infer_schema(records)

    chain_input = {
        "schema": inferred_schema,
        "sample": records[:2]
    }

    ai_msg = chat_model.invoke(prompt.format(**chain_input))
    print(ai_msg.content)
    
    #return llm.invoke(prompt.format(**chain_input))

run_agent("example.json")
#################################################################################################################

from langchain_core.agent import initialize_agent, AgentType

def use_agent():
    json_data = load_json("example.json")
    agent = initialize_agent(
        tools=[json_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    query = f"""
    You are a data architect.

    1. Use the JSONSchemaAnalyzer tool to analyze the dataset.
    2. Based on the tool output, explain the dataset.
    3. Generate a normalized PostgreSQL schema.

    Dataset:
    {json_data}
    """

    response = agent.run(query)
    print(response)

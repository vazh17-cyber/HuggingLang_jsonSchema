from dotenv import load_dotenv
import getpass
import os

load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
elif not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

#########################################################################################################################

from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

#########################################################################################################################

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

# (
#     repo_id="deepseek-ai/DeepSeek-R1-0528",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
#     provider="auto",  # let Hugging Face choose the best provider for you
# )

chat_model = ChatHuggingFace(llm=llm)

#########################################################################################################################

# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# llm = HuggingFacePipeline.from_model_id(
#     model_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )

# chat_model = ChatHuggingFace(llm=llm)

from langchain.messages import (
    HumanMessage,
    SystemMessage,
)

messages = [
    SystemMessage(content="You're a helpful assistant"),
    HumanMessage(
        content="What happens when an unstoppable force meets an immovable object?"
    ),
]

ai_msg = chat_model.invoke(messages)
print(ai_msg.content)

# According to an old philosophical paradox, when an unstoppable force meets an immovable object, 
# it is impossible to determine which one will give way because the concepts themselves contain contradictory meanings. 
# An unstoppable force is something that cannot be stopped, while an immovable object is something that cannot be moved. 
# If we take these definitions literally, a direct collision between the two would result in a paradox, 
# and neither the force nor the object would be able to prevail.

# However, in practical terms, it is important to note that these concepts are often used metaphorically 
# to describe situations where one thing is determined to continue, while another thing is resisting it with all its might. 
# In such cases, the outcome depends on the specific circumstances and the nature of the force and the object involved. 
# For example, a large boulder may be considered an immovable object, but a determined person with the right tools and enough time could move it. 
# Similarly, a river may be considered an unstoppable force, but it can be dammed or diverted.

# So, while the philosophical paradox of an unstoppable force meeting an immovable object is intriguing, 
# it is essential to remember that the real world is more complex, 
# and the outcome of a collision between two seemingly contradictory concepts depends on the specific circumstances and 
# the nature of the force and the object involved.
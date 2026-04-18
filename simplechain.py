from langchain_community.llms import CTransformers
from langchain_core.prompts import PromptTemplate

#cau hinh
model_path = "models/vinallama-7b-chat_q5_0.gguf"
#Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        temperature=0.01,
        max_new_tokens=1024,
        config={
            "stop": ["<|im_end|>"]  # dừng khi gặp token này
        }

    )
    return llm

#tao prompt
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["question"])
    return prompt

#tao chain
def create_simple_chain(prompt, llm):
    chain = prompt | llm
    return chain
#chay thu template
template = """"<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_llm(model_path)
llm_chain = create_simple_chain(prompt, llm)

question = "1+1 bằng bao nhiêu?"
response = llm_chain.invoke({"question": question})
print(response)

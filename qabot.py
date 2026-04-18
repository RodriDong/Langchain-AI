from langchain_community.llms import CTransformers
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
#cau hinh
model_path = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

#Load LLM
def load_llm(model_file):
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        temperature=0.01,
        max_new_tokens=1024,
        config={
            "context_length": 2048,
            "stop": ["<|im_end|>"]
        }

    )
    return llm

#tao prompt
def create_prompt(template):
    prompt = PromptTemplate(template = template, input_variables=["context", "input"])
    return prompt

#tao chain
def create_qa_chain(prompt, llm, db):
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever(search_kwargs={"k": 3}, max_token_limit=1024)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    return chain

#Doc tu vectorDB
def read_vector_db():
    #embeddings
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model,
    allow_dangerous_deserialization=True)
    return db

#thu no
db = read_vector_db()
llm = load_llm(model_path)
template = """<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác. Không trả lời những câu hỏi không liên quan đến nội dung đã cho.
<|im_end|>
<|im_start|>user
Dựa trên nội dung sau đây, hãy trả lời câu hỏi của người dùng. Nếu không có thông tin liên quan, hãy trả lời "Xin lỗi, tôi không biết".
{context}
<|im_end|>
<|im_start|>user
{input}<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)

#chay thu template
question = "protocol là gì?"
response = llm_chain.invoke({"input": question})
print(response["answer"])

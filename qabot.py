from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from tools import grammar_checker, vocabulary_analyzer, coherence_checker, score_calculator, grammar_reference_search
from memory import WritingMemory
from dotenv import load_dotenv
import re
import os

# Doc bien moi truong tu file .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
vector_db_path = "vectorstores/db_faiss"

# Khoi tao bo nho
memory = WritingMemory()

# Load mo hinh ngon ngu qua Groq API
def load_llm():
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )
    return llm

# Doc vector db tu local

#def read_vector_db():
    #embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    #db = FAISS.load_local(
    #    vector_db_path,
   #     embedding_model,
  #      allow_dangerous_deserialization=True
 #   )
#    return db

# Tao ReAct agent voi cac tools cham writing
def create_writing_agent(llm, memory: WritingMemory, user_context: str = ""):
    tools = [grammar_checker, vocabulary_analyzer, coherence_checker, score_calculator,  grammar_reference_search]

    # Lay lich su chi tiet de inject vao reasoning
    history_detail = ""
    if memory.history:
        history_detail = "Previous evaluations:\n"
        for i, entry in enumerate(memory.history[-3:]):  # Lay 3 bai gan nhat
            history_detail += (
                f"- Submission {i+1}: Score {entry.get('score', 'N/A')}/10. "
                f"Writing: {entry['writing_snippet'][:100]}... "
                f"Feedback: {entry['feedback_summary'][:150]}...\n"
            )
    else:
        history_detail = "No previous evaluations available."

    system_prompt = f"""You are a professional English writing evaluator.

User info: {user_context}

{history_detail}

IMPORTANT INSTRUCTIONS:
1. Use the history above to personalize your feedback. For example:
   - If the user previously had grammar issues, pay extra attention to grammar this time
   - If the user has improved since last time, acknowledge the improvement
   - Adjust difficulty of feedback based on user level
2. You MUST use at least 2 tools before giving your final answer
3. All feedback must be written in Vietnamese with full diacritical marks

After using the tools, structure your response as:

## Kết quả chấm bài

### Điểm số
- Ngữ pháp: X/10
- Từ vựng: X/10
- Mạch lạc: X/10
- Nội dung: X/10
- Tổng điểm: X/10

### Nhận xét chi tiết
[Nhận xét cụ thể cho từng tiêu chí]

### So sánh với bài trước
[So sánh điểm mạnh/yếu với lần chấm trước nếu có]

### Điểm mạnh
[Những gì người viết làm tốt]

### Cần cải thiện
[Gợi ý cụ thể]"""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=memory.get_memory()
    )

    return agent


# Chay thu
if __name__ == "__main__":

    llm = load_llm()
    agent = create_writing_agent(llm, memory)

    writing = """
    Yesterday I go to the market with my mother. We buyed many vegetable and fruit.
    The market was very crowd and noise. I like go to market because can see many thing.
    My mother always say that fresh food is more good than food in supermarket.
    """

    # thread_id de langgraph nhan biet phien lam viec
    config = {"configurable": {"thread_id": "session_1"}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": writing}]},
        config=config
    )
    result = response["messages"][-1].content
    print(result)

    # Trich xuat diem tong va luu vao lich su
    score_match = re.search(r'Tổng điểm.*?(\d+(?:\.\d+)?)/10', result)
    score = float(score_match.group(1)) if score_match else 5.0
    memory.save_evaluation(writing, result, score)
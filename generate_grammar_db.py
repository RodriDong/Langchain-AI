from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# Danh sach chu de ngu phap can tao
grammar_topics = [
    "Simple Present Tense",
    "Simple Past Tense",
    "Present Continuous Tense",
    "Past Continuous Tense",
    "Present Perfect Tense",
    "Irregular verbs",
    "Countable and uncountable nouns",
    "Adjectives and adverbs",
    "Comparatives and superlatives",
    "Articles: a, an, the",
    "Prepositions",
    "Conjunctions and transition words",
    "Subject verb agreement",
    "Sentence structure",
    "Common grammar mistakes"
]

def generate_grammar_docs():
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

    documents = []
    for topic in grammar_topics:
        print(f"Đang tạo tài liệu: {topic}...")
        response = llm.invoke(
            f"Explain the grammar rule for '{topic}' in detail. "
            f"Include: definition, usage rules, examples of correct and incorrect usage, "
            f"and common mistakes to avoid. Keep it concise but comprehensive."
        )
        doc = Document(
            page_content=response.content,
            metadata={"topic": topic}
        )
        documents.append(doc)

    return documents

def create_grammar_db():
    # Tao tai lieu ngu phap bang AI
    documents = generate_grammar_docs()

    # Tao embeddings va luu vao FAISS
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local("vectorstores/db_faiss")
    print("Tao vector DB thanh cong!")

create_grammar_db()
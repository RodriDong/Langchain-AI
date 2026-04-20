from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.tools import tool
import json

# Khởi tạo vector DB một lần duy nhất
def load_vector_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(
        "vectorstores/db_faiss",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

db = load_vector_db()
@tool
def grammar_checker(text: str) -> str:
    """Check grammar issues in the writing. Input is the full writing text."""
    words = text.split()
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    avg_sentence_length = round(len(words) / len(sentences), 1) if sentences else 0

    return json.dumps({
        "total_words": len(words),
        "total_sentences": len(sentences),
        "avg_sentence_length": avg_sentence_length,
        "note": "Check for grammar issues in the writing above"
    })
@tool
def grammar_reference_search(query: str) -> str:
    """Search grammar rules and references from the knowledge base. Input is a grammar topic or error type."""
    results = db.similarity_search(query, k=2)
    references = [doc.page_content for doc in results]
    return json.dumps({
        "query": query,
        "references": references
    })

@tool
def vocabulary_analyzer(text: str) -> str:
    """Analyze vocabulary richness and word choice. Input is the full writing text."""
    words = text.split()
    unique_words = set(w.lower().strip('.,!?;:') for w in words)
    word_count = len(words)
    unique_count = len(unique_words)
    ratio = round(unique_count / word_count * 100, 1) if word_count > 0 else 0
    avg_word_length = round(sum(len(w) for w in words) / word_count, 1) if word_count > 0 else 0

    return json.dumps({
        "total_words": word_count,
        "unique_words": unique_count,
        "lexical_diversity": f"{ratio}%",
        "avg_word_length": avg_word_length
    })

@tool
def coherence_checker(text: str) -> str:
    """Check structure, coherence and flow of the writing. Input is the full writing text."""
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    # Kiểm tra transition words
    transition_words = ["however", "therefore", "furthermore", "moreover", "in addition",
                        "consequently", "nevertheless", "although", "because", "since",
                        "first", "second", "finally", "in conclusion", "for example"]
    found_transitions = [w for w in transition_words if w.lower() in text.lower()]

    return json.dumps({
        "paragraph_count": len(paragraphs),
        "sentence_count": len(sentences),
        "avg_sentences_per_paragraph": round(len(sentences) / len(paragraphs), 1) if paragraphs else 0,
        "transition_words_found": found_transitions,
        "transition_word_count": len(found_transitions)
    })

@tool
def score_calculator(grammar: float, vocabulary: float, coherence: float, content: float) -> str:
    """Calculate overall writing score. Input: grammar, vocabulary, coherence, content scores (0-10)."""
    overall = round((grammar * 0.3 + vocabulary * 0.25 + coherence * 0.25 + content * 0.2), 1)

    if overall >= 9:
        band = "Excellent"
    elif overall >= 7:
        band = "Good"
    elif overall >= 5:
        band = "Adequate"
    elif overall >= 3:
        band = "Developing"
    else:
        band = "Needs Improvement"

    return json.dumps({
        "grammar_score": grammar,
        "vocabulary_score": vocabulary,
        "coherence_score": coherence,
        "content_score": content,
        "overall_score": overall,
        "band": band
    })
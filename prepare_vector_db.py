from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Cấu hình đường dẫn
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def create_db_from_files():
    # Quét toàn bộ file PDF trong thư mục data
    loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Chia nhỏ văn bản thành các đoạn
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Tạo embeddings và lưu vào FAISS
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_files()
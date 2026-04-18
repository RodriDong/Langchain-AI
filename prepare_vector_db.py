from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

#khai bao
pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

#Tao ra vectorDB tu 1 doan van ban
def create_db_from_text():
    raw_text = """Nhằm đáp ứng nhu cầu và thị hiếu của khách hàng về việc sở hữu số tài khoản đẹp, dễ nhớ, 
    giúp tiết kiệm thời gian, mang đến sự thuận tiện và may mắn trong kinh doanh. 
    Cụ thể, đối với tài khoản số đẹp 9 số, SHB miễn phí mở tài khoản số đẹp trị giá 880.000đ; 
    giảm tới 80% phí mở tài khoản số đẹp trị giá từ 1,1 triệu đồng trở lên. 
    Đối với tài khoản số đẹp 12 số, SHB miễn 100% phí mở tài khoản số đẹp, khách hàng có thể lựa chọn 
    tối đa toàn bộ dãy số của tài khoản. Đây là một thế mạnh giúp SHB thu hút khách hàng.
    Hiện nay, SHB đang cung cấp đến khách hàng 3 loại tài khoản số đẹp: 9 số, 10 số và 12 số. 
    Cùng với sự tiện lợi khi giao dịch online mọi lúc mọi nơi trên nền tảng ngân hàng số.
    Ngoài kênh giao dịch tại quầy, khách hàng cũng dễ dàng mở tài khoản số đẹp trên ứng dụng SHB Mobile 
    mà không cần hồ sơ thủ tục phức tạp, giúp tối ưu hóa trải nghiệm người dùng.
    Hướng tới mục tiêu trở thành ngân hàng số 1 về hiệu quả tại Việt Nam, ngân hàng bán lẻ hiện đại nhất 
    và là ngân hàng số được yêu thích nhất tại Việt Nam. 
    Để biết thêm thông tin về chương trình, Quý khách vui lòng liên hệ các điểm giao dịch của SHB 
    trên toàn quốc hoặc Hotline *6688."""

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len

    )
    chunks = text_splitter.split_text(raw_text)

    #Embeddings
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")

    # Đưa vào Faiss VectorDB
    db = FAISS.from_texts(texts = chunks, embedding = embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_files():
    #khai bao loader de quet toan bo thu muc
    loader = DirectoryLoader(pdf_data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunk = text_splitter.split_documents(documents)

    #Embeddings
    embedding_model = GPT4AllEmbeddings(model_file = "models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunk, embedding_model)
    db.save_local(vector_db_path)
    return db

create_db_from_files()

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
    Docx2txtLoader,
)
from pathlib import Path
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
#from transformers import AutoTokenizer, AutoModel

#tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def langchain_document_loader(UPLOAD_FOLDER):

    documents = []

    txt_loader = DirectoryLoader(
        UPLOAD_FOLDER, glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    documents.extend(txt_loader.load())

    pdf_loader = DirectoryLoader(
        UPLOAD_FOLDER.as_posix(), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    csv_loader = DirectoryLoader(
        UPLOAD_FOLDER.as_posix(), glob="**/*.csv", loader_cls=CSVLoader, show_progress=True,
        loader_kwargs={"encoding":"utf8"}
    )
    documents.extend(csv_loader.load())

    doc_loader = DirectoryLoader(
        UPLOAD_FOLDER.as_posix(),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        show_progress=True,
    )
    documents.extend(doc_loader.load())
    return documents

def split_documents_to_chunks(documents):
    """Split documents to chunks using RecursiveCharacterTextSplitter."""

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def select_embeddings_model(key):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key= key, model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    #embeddings = GoogleGenerativeAIEmbeddings(
       # model="models/embedding-001", google_api_key="AIzaSyAVwUcWcZB06BXBEbZa60DX9rmb8rYgZdo"
    #)
    # embeddings = HuggingFaceInferenceAPIEmbeddings(    
    #         api_key="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",
    #         model_name="sentence-transformers/all-MiniLM-L6-v2" 
    #         # model_name="thenlper/gte-large"
    #     )
    return embeddings

def create_vectorDB(chunks,embeddings,Vector_DB_Name,VECTOR_STORE):
    # print(str(VECTOR_STORE)+"/"+"ChromaDB", start = "\n\n", end = "\n\n")
    vector_store = Chroma.from_documents(
                            documents=chunks,
                            embedding=embeddings,
                            persist_directory= str(VECTOR_STORE)+"/"+"ChromaDB",
                        )
    return vector_store
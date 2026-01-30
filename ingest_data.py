import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
DATA_DIR = "data/"
DB_PATH = "faiss_index"

def update_knowledge_base():
    print(f"--- Scanning {DATA_DIR} for all knowledge ---")
    
    # This part loads EVERY .txt file in the folder
    loader = DirectoryLoader(DATA_DIR, glob="./*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("Error: No .txt files found in the data folder!")
        return

    # Split and Convert
    text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    print(f"--- Processing {len(docs)} chunks of data ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Save the new Library
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(DB_PATH)
    print("🚀 SUCCESS! Your brain is now smarter and includes all files.")

if __name__ == "__main__":
    update_knowledge_base()
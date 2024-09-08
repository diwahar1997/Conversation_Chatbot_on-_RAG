from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
import os

class Data_ingestion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.embeddings = HuggingFaceEmbeddings(model_name=model) 
       

    def read_pdf_load_documents(self):
        directory = self.file_path
        loaded_documents = []
        files = [file for file in os.listdir(directory) if file.endswith('.csv')]
        for file in files:
            loader = CSVLoader(os.path.join(directory,file),encoding='utf-8')
            documents = loader.load()
            loaded_documents.extend(documents)
        return loaded_documents

    def Split_documents(self, file_data):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        final_documents = text_splitter.split_documents(file_data)
        return final_documents

    def vector_data_base(self, final_documents):
        vectors_db = FAISS.from_documents(final_documents, self.embeddings)
        vectors_db.save_local("trail_db")

    def process_documents(self):
        file_data = self.read_pdf_load_documents()
        final_documents = self.Split_documents(file_data)
        self.vector_data_base(final_documents)
        print("Database_Created")

model='all-MiniLM-L6-v2'
def main():
    file_path = r"docs"
    processor = Data_ingestion(file_path)
    processor.process_documents()

if __name__ == "__main__":
    main()

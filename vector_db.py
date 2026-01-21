from langchain_pinecone import PineconeVectorStore
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from pinecone import ServerlessSpec
import os
from dotenv import load_dotenv
import glob
import re
load_dotenv()

class VectorDB:
    def __init__(self, index_name,embedding_model= "llama-text-embed-v2"):
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings=PineconeEmbeddings(model=embedding_model)

        self._initialize_vector_store()

    def _initialize_vector_store(self):
        # Check if index exists
        if not self.pc.has_index(self.index_name):
            print(f"Creating index '{self.index_name}'...")
            # create the index
            index_model = self.pc.create_index(
                name=self.index_name,
                dimension=1024,  # Dimension for 'llama-text-embed-v2'
                metric="cosine",
                spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1")
            )
            # establish an index connection
            self.index = self.pc.Index(host=index_model.host)
            
        else:
            # Ensure host is resolved even if index already exists
            try:
                index_description = self.pc.describe_index(self.index_name)
                self.host = index_description.host
                self.index= self.pc.Index(host=self.host)
                print(
                    f"Index '{self.index_name}' already exists. Using host: {self.host}"
                )
            except Exception as e:
                print(f"Error resolving host in existing index: {e}")

        # LangChain vector store wrapper
        self.vector_store = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings
        )

    def process_docs(self, path: str):
        all_chunks=[]
        pdf_files = glob.glob(f"{path}/*.pdf")
        
        for file in pdf_files:

            loader = PyPDFLoader(file)
            documents = loader.load()
            full_text = " ".join([doc.page_content for doc in documents])
                
            # Extract Model Number using Regex
            model_match = re.search(r"Model:\s*(.*)", full_text)
            model_name = model_match.group(1).strip() if model_match else "Unknown Model"
            # Chunk Documents 
            # Use RecursiveCharacterTextSplitter for robust splitting that respects document structure
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150, # Overlap helps maintain context between chunks
                
            )
            doc_chunks = text_splitter.split_documents(documents)

            #Add Product Model metadata to chunks
            for chunk in doc_chunks:
                chunk.metadata["model_name"] = model_name
                # Also useful to keep the source filename
                chunk.metadata["source"] = file
                
                all_chunks.extend(doc_chunks)
                print(f"Processed {file}: Extracted model '{model_name}'")
                # Print the number of chunks after splitting
        print(f"After splitting: {len(all_chunks)} chunks")
        return all_chunks
        

    def add_documents(self, documents):
        self.vector_store.add_documents(
        documents=documents,
        namespace="policy_documents"
        )
        print(f"Added {len(documents)} documents to the vector store.")

    def process_upsert(self, path: str):
        documents = self.process_docs(path)
        self.add_documents(documents)
        

    def query(self, query_text, top_k=5, filters:dict=None):
        matches = self.vector_store.similarity_search(
            query_text,
            k=top_k,
            namespace="policy_documents",
            filter=filters
        )
        print(f"Retrieved {len(matches)} matching documents.")
        return matches


    
if __name__ == "__main__":
    vector_db = VectorDB()
    vector_db.process_upsert("BreezeLite_Professional_Manuals/")
    results = vector_db.query("Is water immersion damage covered under warranty for BreezeLite BLD 150 hair dryers?", filters={"source": {"$eq": "BreezeLite_Professional_Manuals/BreezeLite_BLD-150_Everyday.pdf"}})
    print("Query Results:", results)
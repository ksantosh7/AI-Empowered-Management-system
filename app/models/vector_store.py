
# import chromadb
# from langchain.vectorstores import Chroma
# from langchain.embeddings.openai import OpenAIEmbeddings

# class VectorStore:
#     def __init__(self, path):
#         self.embeddings = OpenAIEmbeddings()
#         self.vector_store = Chroma(
#             persist_directory=path,
#             embedding_function=self.embeddings
#         )

#     def add_documents(self, documents):
#         self.vector_store.add_documents(documents)
        
#     def similarity_search(self, query, k=4):
#         return self.vector_store.similarity_search(query, k=k)

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings


class PineconeVectorStore:
    def __init__(self, index_name, pinecone_api_key, pinecone_env):
        # Initialize Pinecone client
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        
        # Connect to or create the index with specified dimension
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            # You need to set dimension according to your embeddings size, typically 1536 for OpenAI embeddings
            pinecone.create_index(index_name, dimension=1536, metric="cosine")
        
        # Load OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize LangChain Pinecone wrapper
        self.vector_store = Pinecone(
            index=pinecone.Index(index_name),
            embedding_function=self.embeddings.embed_query,
            text_key="text"
        )
    
    def add_documents(self, documents):
        # documents should be a list of LangChain Document objects or dicts with 'text' field
        self.vector_store.add_documents(documents)
    
    def similarity_search(self, query, k=4):
        # Perform similarity search, returns list of top k matching documents
        return self.vector_store.similarity_search(query, k=k)

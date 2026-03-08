"""
Medical Retriever
Handles document retrieval from Pinecone vector store.
Provides optimized retrieval for different agent types.
"""

from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
import os


class MedicalRetriever:
    def __init__(self, index_name="medical-chatbot", search_k=3):
        """
        Initialize Medical Retriever
        
        Args:
            index_name: Name of the Pinecone index
            search_k: Number of documents to retrieve (default: 3)
        """
        self.index_name = index_name
        self.search_k = search_k
        
        # Load embeddings
        self.embeddings = download_hugging_face_embeddings()
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.search_k}
        )
    
    def get_retriever(self, k=None):
        """
        Get the retriever object
        
        Args:
            k: Number of documents to retrieve (optional, uses default if not provided)
            
        Returns:
            Retriever object
        """
        if k is not None and k != self.search_k:
            return self.vectorstore.as_retriever(search_kwargs={"k": k})
        return self.retriever
    
    def retrieve_documents(self, query, k=None):
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve (optional)
            
        Returns:
            List of Document objects
        """
        if k is not None:
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
            return retriever.invoke(query)
        return self.retriever.invoke(query)
    
    def retrieve_with_scores(self, query, k=None):
        """
        Retrieve documents with relevance scores
        
        Args:
            query: Search query
            k: Number of documents to retrieve (optional)
            
        Returns:
            List of tuples (Document, score)
        """
        k = k or self.search_k
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def get_medical_retriever(self):
        """
        Get retriever optimized for medical knowledge queries
        
        Returns:
            Retriever configured for medical queries
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def get_symptom_retriever(self):
        """
        Get retriever optimized for symptom queries
        (retrieves more documents for better context)
        
        Returns:
            Retriever configured for symptom queries
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": 4})
    
    def get_lifestyle_retriever(self):
        """
        Get retriever optimized for lifestyle/wellness queries
        
        Returns:
            Retriever configured for lifestyle queries
        """
        return self.vectorstore.as_retriever(search_kwargs={"k": 3})

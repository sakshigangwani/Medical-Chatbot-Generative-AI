import os
from dotenv import load_dotenv

from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def main():
    load_dotenv()

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set")

    pc = Pinecone(api_key=api_key)
    index_name = "medical-chatbot"

    # load PDF documents from Data directory relative to this script
    pdf_dir = os.path.join(os.path.dirname(__file__), "Data")
    documents = load_pdf(pdf_dir)

    # split into chunks
    text_chunks = text_split(documents)

    # prepare embeddings
    embeddings = download_hugging_face_embeddings()

    # create index if missing
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    # upsert the documents
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings
    )

    print("Embeddings stored successfully in Pinecone!")


if __name__ == "__main__":
    main()
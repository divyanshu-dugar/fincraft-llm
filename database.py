import os
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# --- Initialize MongoDB and Vector Store ---
mongo_uri = os.getenv("MONGODB_URL")
if not mongo_uri:
    raise ValueError("MONGODB_URL environment variable is not set")

client = MongoClient(mongo_uri)
expense_collection = client["user-data"].expenses
logger.info("✅ MongoDB connection established")

# Initialize OpenAI embeddings for semantic search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize MongoDB Atlas Vector Search
expense_vector_store = MongoDBAtlasVectorSearch(
    collection=expense_collection,
    embedding=embeddings,
    index_name="fincraft_vector_index",  
    text_key="note",  
    embedding_key="embedding"  
)
logger.info("✅ Vector store initialized")

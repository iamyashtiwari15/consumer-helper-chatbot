from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGODB_DB_NAME", "consumer_chatbot")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "conversations")

# Async MongoDB client for FastAPI
async def get_mongodb_client():
    client = AsyncIOMotorClient(MONGODB_URI)
    try:
        await client.admin.command('ping')
        print("✅ Connected to MongoDB!")
        return client
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        raise

# Async MongoDB initialization
async def init_mongodb():
    client = AsyncIOMotorClient(MONGODB_URI)
    db = client[DB_NAME]
    
    try:
        # Check if collection exists
        collections = await db.list_collection_names()
        if COLLECTION_NAME not in collections:
            await db.create_collection(COLLECTION_NAME)
            
            # Create indexes
            collection = db[COLLECTION_NAME]
            await collection.create_index("conversation_id", unique=True)
            await collection.create_index("created_at")
            await collection.create_index("updated_at")
            print(f"✅ Collection '{COLLECTION_NAME}' initialized with indexes!")
        
        print("✅ MongoDB initialized successfully!")
        return client
    except Exception as e:
        print(f"❌ Error initializing MongoDB: {e}")
        raise

# Get database and collection
async def get_db():
    client = await get_mongodb_client()
    return client[DB_NAME]

async def get_collection():
    db = await get_db()
    return db[COLLECTION_NAME]

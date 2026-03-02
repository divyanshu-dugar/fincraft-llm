import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the router directly
from routes import router

app = FastAPI(
    title="Fincraft AI Service",
    description="AI-powered financial insights using GPT-4o and vector search",
    version="1.0.0"
)

# Configure CORS to allow requests from frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS middleware configured")

# Register all API endpoints
app.include_router(router)

if __name__ == '__main__':
    import uvicorn
    
    # Read port from environment or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Start uvicorn server
    logger.info(f"Starting Fincraft AI Service on port {port}")
    logger.info(f"📖 API Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(
        "main:app",  # Pointing to this file (main.py)
        host="0.0.0.0",
        port=port,
        reload=True
    )

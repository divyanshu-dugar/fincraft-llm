"""
Fincraft AI Service - Financial AI Assistant
==============================================

A FastAPI-based microservice that provides AI-powered financial insights
using LangChain, OpenAI's GPT-4o model, and MongoDB Atlas Vector Search.

Key Features:
    - Semantic search of expenses using vector embeddings
    - AI-powered financial advice with function calling
    - Expense trend analysis and insights
    - RESTful API with proper error handling

Architecture:
    - FastAPI: Web framework for REST API
    - LangChain: LLM orchestration and tool binding
    - MongoDB Atlas: Vector database for semantic search
    - OpenAI: GPT-4o for conversation and embeddings

Author: Divyanshu Dugar
Version: 1.0.0
"""

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dateutil import parser
from typing import List, Dict, Any

from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from bson import ObjectId

# Load environment variables from .env file
load_dotenv()

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# INITIALIZATION & CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Fincraft AI Service",
    description="AI-powered financial insights using GPT-4o and vector search",
    version="1.0.0"
)

# --- Initialize MongoDB and Vector Store ---
# Connect to MongoDB Atlas to access the user-data database
mongo_uri = os.getenv("MONGODB_URL")
if not mongo_uri:
    raise ValueError("MONGODB_URL environment variable is not set")

client = MongoClient(mongo_uri)
expense_collection = client["user-data"].expenses
logger.info("✅ MongoDB connection established")

# Initialize OpenAI embeddings for semantic search
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Initialize MongoDB Atlas Vector Search
# This enables semantic similarity search on expense descriptions
expense_vector_store = MongoDBAtlasVectorSearch(
    collection=expense_collection,
    embedding=embeddings,
    index_name="fincraft_vector_index",  
    text_key="note",  # Field containing expense description
    embedding_key="embedding"  # Field containing vector embeddings
)
logger.info("✅ Vector store initialized")

# ============================================================================
# TOOLS & LLM CONFIGURATION
# ============================================================================

def search_expenses(query_text: str, user_id: str) -> str:
    """
    Search user expenses using semantic similarity and vector embeddings.
    
    This function performs a vector similarity search on the user's expenses
    to find relevant transactions based on semantic meaning rather than
    exact keyword matching. For example, searching "coffee" will find
    expenses with descriptions like "cafe latte", "espresso bar", etc.
    
    The search is pre-filtered by user_id at the database level to ensure
    users only see their own expenses (privacy & security).
    
    Args:
        query_text (str): Semantic search query describing the expense
                         (e.g., "coffee", "lunch", "transportation")
        user_id (str): MongoDB ObjectId of the user as a string
    
    Returns:
        str: Formatted string containing matching expenses with amounts, dates,
             and descriptions. Returns "No matching expenses found." if no
             results match the query.
             
             Format: "Amount: $X, Date: YYYY-MM-DD, Note/Desc: Description"
    
    Raises:
        Exception: Caught and logged; returns error message string
    
    Example:
        >>> search_expenses("rent", "68dc575441c88ece4665c47d")
        "Amount: $500, Date: 2026-03-01, Note/Desc: Rent - Reena Mam"
    """
    try:
        logger.debug(f"Searching expenses: query='{query_text}', user_id='{user_id}'")
        
        # Use pre_filter to filter by user_id at MongoDB level (more efficient)
        # This ensures data privacy and reduces unnecessary processing
        results = expense_vector_store.similarity_search(
            query=query_text,  # Semantic search query
            k=5,  # Return top 5 most relevant results
            pre_filter={"user": ObjectId(user_id)}  # Filter by user_id
        )
        
        if not results:
            logger.info(f"No expenses found for user {user_id} matching: {query_text}")
            return "No matching expenses found."
        
        # Format results for display to user
        formatted_results = []
        for doc in results:
            amt = doc.metadata.get('amount', 'Unknown')
            date = doc.metadata.get('date', 'Unknown Date')
            note = doc.page_content if doc.page_content else 'No note'
            formatted_results.append(f"Amount: ${amt}, Date: {date}, Note/Desc: {note}")
        
        logger.info(f"Found {len(formatted_results)} results for user {user_id}")
        return "\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Error searching expenses for user {user_id}: {str(e)}", exc_info=True)
        return f"Error searching expenses: {str(e)}"

# Create tool object for LLM function calling
# The @tool decorator makes this function callable by the LLM
search_expenses_tool = tool(search_expenses)

# Initialize ChatGPT LLM instance with specified configuration
llm = ChatOpenAI(
    model="gpt-4o",  # Latest GPT-4 model
    temperature=0.7  # Balanced creativity (0.0=factual, 1.0=creative)
)
tools = [search_expenses_tool]
logger.info("✅ LLM initialized")

# System prompt that guides LLM behavior
# This prompt constrains the AI to financial topics and enables tool usage
SYSTEM_PROMPT = """
You are Fincraft AI - a highly capable personal finance mentor. 
You ONLY answer questions related to personal finance (income, saving goals, expense, and budgets).
Exclude anything related to investments. If the user asks something other than money management, politely refuse.

You have access to tools to look up the user's secure financial data. Always use these tools if the user asks about their own spending.
"""

# Bind tools to the LLM for function calling
# This allows the LLM to decide when to call the search_expenses tool
llm_with_tools = llm.bind_tools(tools)
logger.info("✅ LLM tools bound successfully")

# ============================================================================
# MIDDLEWARE & CONFIGURATION
# ============================================================================

# Configure CORS to allow requests from frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
logger.info("✅ CORS middleware configured")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class UserQuery(BaseModel):
    """
    Request model for user financial queries.
    
    Attributes:
        userQuery (str): The user's financial question or request
        userId (str): Unique identifier of the user (MongoDB ObjectId as string)
    """
    userQuery: str = Field(..., description="User's financial question")
    userId: str = Field(..., description="User's unique ID")

@app.post('/api/ai/chat')
async def chat(query: UserQuery):
    """
    Process user financial query and return AI-generated response.
    
    This endpoint implements a two-step conversation loop:
    1. First LLM call: Determines if tools need to be called for data
    2. Tool execution: If needed, searches user's expense data
    3. Second LLM call: Generates polished response based on tool results
    
    The LLM decides whether to use the search_expenses tool based on the
    user's question. For example:
    - "What did I spend on rent?" → Uses search_expenses tool
    - "What is budgeting?" → Answers directly without tools
    
    Args:
        query (UserQuery): Request containing userQuery and userId
    
    Returns:
        str: AI-generated response to the user's query
    
    Raises:
        HTTPException: 500 error if processing fails
    
    Example Request:
        POST /api/ai/chat
        {
            "userQuery": "What did I spend on rent this month?",
            "userId": "68dc575441c88ece4665c47d"
        }
    
    Example Response:
        "Based on your expense data, you spent $500 on rent on March 1st.
         This represents 25% of your monthly spending..."
    """
    try:
        logger.info(f"Chat query from user {query.userId}: {query.userQuery[:100]}")
        
        # Step 1: Build message chain with system prompt and user query
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User ID: {query.userId}\n\n{query.userQuery}")
        ]
        
        # Step 2: Call LLM with tools bound
        # The LLM will decide if it needs to call search_expenses
        response = llm_with_tools.invoke(messages)
        logger.debug(f"LLM response type: {type(response).__name__}")
        
        # Step 3: Check if LLM wants to call tools
        if response.tool_calls:
            logger.info(f"Tool calls detected: {[tc['name'] for tc in response.tool_calls]}")
            
            # Add the assistant's response to message history
            messages.append(response)
            
            # Process each tool call
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_expenses':
                    # Extract arguments from tool call
                    query_text = tool_call['args'].get('query_text', '')
                    user_id_arg = tool_call['args'].get('user_id', query.userId)
                    logger.debug(f"Executing search_expenses: query='{query_text}'")
                    
                    # Call the actual search function
                    result = search_expenses(
                        query_text=query_text,
                        user_id=user_id_arg
                    )
                    logger.debug(f"Tool result: {result[:100]}")
                    
                    # Add tool result to message chain
                    # This follows OpenAI's function calling protocol
                    messages.append(ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=result
                    ))
            
            # Step 4: Call LLM again to generate polished response
            # The LLM now has the expense data and can craft a better answer
            final_response = llm.invoke(messages)
            logger.info(f"Generated polished response")
            return final_response.content
        
        # No tools needed - return direct LLM response
        logger.info("No tools needed, returning direct response")
        return response.content
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process AI chat")

class AnalyzeRequest(BaseModel):
    """
    Request model for expense trend analysis.
    
    Attributes:
        expenses (List[Dict]): List of expense documents with date and amount
    """
    expenses: List[Dict[str, Any]] = Field(..., description="List of expenses to analyze")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post('/api/analyze')
def analyze_expenses(body: AnalyzeRequest):
    """
    Analyze expense trends and generate insights.
    
    This endpoint processes a list of expenses, groups them by month,
    calculates month-over-month changes, and generates human-readable
    insights about spending patterns.
    
    Process:
    1. Parse and validate expense dates
    2. Group expenses by month (YYYY-MM format)
    3. Calculate total spending per month
    4. Determine trend: increased, decreased, or stable
    5. Generate insight message
    
    Args:
        body (AnalyzeRequest): Request containing list of expenses
    
    Returns:
        dict: Dictionary with:
            - data (List): Monthly aggregated expenses
            - insight (str): AI-generated insight about trends
    
    Raises:
        HTTPException: 500 error if analysis fails
    
    Example Request:
        POST /api/analyze
        {
            "expenses": [
                {
                    "_id": "69a59cbad671ed1a85f8eb9b",
                    "amount": 500,
                    "date": "2026-03-01T00:00:00.000Z"
                },
                {
                    "_id": "69a59cbad671ed1a85f8eb9c",
                    "amount": 550,
                    "date": "2026-02-01T00:00:00.000Z"
                }
            ]
        }
    
    Example Response:
        {
            "data": [
                {"month": "2026-02", "amount": 550},
                {"month": "2026-03", "amount": 500}
            ],
            "insight": "Your expenses have decreased by 9.09% compared to last month."
        }
    """
    try:
        expenses = body.expenses
        logger.info(f"Analyzing {len(expenses)} expenses")
        
        # Handle empty expense list
        if not expenses:
            return {
                "data": [],
                "insight": "No expenses found for the selected period."
            }

        # Parse and prepare expenses for analysis
        processed_expenses = []
        for exp in expenses:
            try:
                # Parse ISO date string to datetime object
                parsed_date = parser.parse(exp['date'])
                expense_copy = exp.copy()
                expense_copy['_parsed_date'] = parsed_date
                processed_expenses.append(expense_copy)
            except (KeyError, ValueError, TypeError) as e:
                # Skip malformed expenses but log them
                logger.warning(f"Skipping expense {exp.get('_id', 'unknown')}: {e}")
                continue

        # Sort by parsed date (oldest first)
        processed_expenses.sort(key=lambda x: x['_parsed_date'])

        # Group by month (YYYY-MM format)
        monthly_data = {}
        for exp in processed_expenses:
            d = exp['_parsed_date']
            key = f"{d.year}-{d.month:02d}"
            if key not in monthly_data:
                monthly_data[key] = 0
            monthly_data[key] += exp['amount']

        # Convert to sorted list for response
        monthly_list = [{"month": key, "amount": amt} for key, amt in monthly_data.items()]
        monthly_list.sort(key=lambda x: x['month'])

        # Generate insight based on trend (month-over-month comparison)
        if len(monthly_list) >= 2:
            prev, curr = monthly_list[-2], monthly_list[-1]
            # Calculate percentage change (safe division with zero check)
            if prev["amount"] != 0:
                change = ((curr["amount"] - prev["amount"]) / prev["amount"]) * 100
            else:
                change = 0
            # Determine trend direction
            trend = "increased" if change > 0 else "decreased" if change < 0 else "remained the same"
            insight = f"Your expenses have {trend} by {abs(change):.2f}% compared to last month."
        elif len(monthly_list) == 1:
            # First month of tracking
            insight = "This is your first month of tracking expenses. Keep it up!"
        else:
            # No valid data
            insight = "Not enough data to generate insights."
        
        logger.info(f"Analysis complete: {len(monthly_list)} months of data analyzed")

        return {
            "data": monthly_list,
            "insight": insight
        }

    except Exception as e:
        logger.error(f"Error in analyze_expenses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get('/')
def root():
    """
    Root endpoint - redirects to health check.
    
    Returns:
        RedirectResponse: Redirect to /api/health
    """
    return RedirectResponse("/api/health")


@app.get('/api/health')
def health_check():
    """
    Health check endpoint for monitoring service status.
    
    This endpoint is used by load balancers and monitoring systems
    to verify that the service is running.
    
    Returns:
        dict: Status information with keys:
            - status (str): "healthy" if service is running
            - service (str): Service name
    """
    logger.info("Health check requested")
    return {"status": "healthy", "service": "FastAPI AI Service"}


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import uvicorn
    
    # Read port from environment or use default
    port = int(os.environ.get('PORT', 8000))
    
    # Start uvicorn server
    logger.info(f"Starting Fincraft AI Service on port {port}")
    logger.info("📖 API Documentation: http://localhost:{port}/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Listen on all interfaces
        port=port,
        reload=True  # Auto-reload on code changes (development only)
    )
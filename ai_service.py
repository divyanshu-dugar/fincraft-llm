import logging
from bson import ObjectId
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from database import expense_vector_store

logger = logging.getLogger(__name__)

@tool
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
    """
    try:
        logger.debug(f"Searching expenses: query='{query_text}', user_id='{user_id}'")
        
        # Use pre_filter to filter by user_id at MongoDB level 
        results = expense_vector_store.similarity_search(
            query=query_text,
            k=5,
            pre_filter={"user": ObjectId(user_id)}
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

# Initialize ChatGPT LLM instance with specified configuration
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7
)
tools = [search_expenses]
logger.info("✅ LLM initialized")

# System prompt that guides LLM behavior
SYSTEM_PROMPT = """
You are Fincraft AI - a highly capable personal finance mentor. 
You ONLY answer questions related to personal finance (income, saving goals, expense, and budgets).
Exclude anything related to investments. If the user asks something other than money management, politely refuse.

You have access to tools to look up the user's secure financial data. Always use these tools if the user asks about their own spending.
"""

# Bind tools to the LLM for function calling
llm_with_tools = llm.bind_tools(tools)
logger.info("✅ LLM tools bound successfully")

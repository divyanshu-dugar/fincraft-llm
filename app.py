import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser
from typing import List, Dict, Any

from pymongo import MongoClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from bson import ObjectId

load_dotenv()

app = FastAPI()

# --- Initialize MongoDB and Vector Store ---
mongo_uri = os.getenv("MONGODB_URL") 
client = MongoClient(mongo_uri)
expense_collection = client["user-data"].expenses

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
expense_vector_store = MongoDBAtlasVectorSearch(
    collection=expense_collection,
    embedding=embeddings,
    index_name="fincraft_vector_index", 
    text_key="note", 
    embedding_key="embedding"
)

# --- Define Tools ---
def search_expenses(query_text: str, user_id: str) -> str:
    """Searches a user's expenses based on a meaning, category, or description (e.g., 'coffee', 'groceries', 'fun').
    Always use this when the user asks about their spending history.
    """
    try:
        # Use pre_filter with MongoDB Atlas Vector Search to filter by user_id at database level
        results = expense_vector_store.similarity_search(
            query=query_text, 
            k=5,
            pre_filter={"user": ObjectId(user_id)}
        )
        
        if not results:
            return "No matching expenses found."
            
        formatted_results = []
        for doc in results:
            amt = doc.metadata.get('amount', 'Unknown')
            date = doc.metadata.get('date', 'Unknown Date')
            note = doc.page_content if doc.page_content else 'No note'
            formatted_results.append(f"Amount: ${amt}, Date: {date}, Note/Desc: {note}")
            
        return "\n".join(formatted_results)
    except Exception as e:
        print(f"Exception in search_expenses: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error searching expenses: {str(e)}"

# Create tool object for LLM
search_expenses_tool = tool(search_expenses)

# Initialize LangChain LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
tools = [search_expenses_tool]

SYSTEM_PROMPT = """
You are Fincraft AI - a highly capable personal finance mentor. 
You ONLY answer questions related to personal finance (income, saving goals, expense, and budgets).
Exclude anything related to investments. If the user asks something other than money management, politely refuse.

You have access to tools to look up the user's secure financial data. Always use these tools if the user asks about their own spending.
"""

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    userQuery: str
    userId: str

@app.post('/api/ai/chat')
async def chat(query: UserQuery):
    try:
        print(f"Chat request - Query: {query.userQuery}, UserId: {query.userId}")
        
        # Build the initial message with system prompt
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User ID: {query.userId}\n\n{query.userQuery}")
        ]
        
        # Call the LLM with tools
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"Tool calls detected: {response.tool_calls}")
            
            # Add the assistant response to messages
            messages.append(response)
            
            # Process tool calls
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_expenses':
                    # Call the actual function implementation
                    query_text = tool_call['args'].get('query_text', '')
                    user_id_arg = tool_call['args'].get('user_id', query.userId)
                    print(f"Calling search_expenses with query_text='{query_text}', user_id='{user_id_arg}'")
                    
                    result = search_expenses(
                        query_text=query_text,
                        user_id=user_id_arg
                    )
                    print(f"Tool result: {result}")
                    
                    # Add tool message with the result
                    messages.append(ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=result
                    ))
            
            # Call LLM again to generate a polished response based on tool results
            final_response = llm.invoke(messages)
            print(f"Final LLM response: {final_response.content}")
            return final_response.content
        
        # If no tool calls, return the text response
        print(f"No tool calls, returning text response: {response.content}")
        return response.content
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to process AI chat")


# ── /api/analyze  (expense trend analysis) ───────────────────────────────────

class AnalyzeRequest(BaseModel):
    expenses: List[Dict[str, Any]]


@app.post('/api/analyze')
def analyze_expenses(body: AnalyzeRequest):
    try:
        expenses = body.expenses

        if not expenses:
            return {
                "data": [],
                "insight": "No expenses found for the selected period."
            }

        # Process expenses without pandas
        processed_expenses = []
        for exp in expenses:
            try:
                parsed_date = parser.parse(exp['date'])
                expense_copy = exp.copy()
                expense_copy['_parsed_date'] = parsed_date
                processed_expenses.append(expense_copy)
            except (KeyError, ValueError, TypeError) as e:
                print(f"Skipping expense {exp.get('_id', 'unknown')}: {e}")
                continue

        # Sort by parsed date
        processed_expenses.sort(key=lambda x: x['_parsed_date'])

        # Group by month-year using standard dict
        monthly_data = {}
        for exp in processed_expenses:
            d = exp['_parsed_date']
            key = f"{d.year}-{d.month:02d}"
            if key not in monthly_data:
                monthly_data[key] = 0
            monthly_data[key] += exp['amount']

        # Convert to list and sort
        monthly_list = [{"month": key, "amount": amt} for key, amt in monthly_data.items()]
        monthly_list.sort(key=lambda x: x['month'])

        # Generate insight
        if len(monthly_list) >= 2:
            prev, curr = monthly_list[-2], monthly_list[-1]
            change = ((curr["amount"] - prev["amount"]) / prev["amount"]) * 100 if prev["amount"] != 0 else 0
            trend = "increased" if change > 0 else "decreased" if change < 0 else "remained the same"
            insight = f"Your expenses have {trend} by {abs(change):.2f}% compared to last month."
        elif len(monthly_list) == 1:
            insight = "This is your first month of tracking expenses. Keep it up!"
        else:
            insight = "Not enough data to generate insights."

        return {
            "data": monthly_list,
            "insight": insight
        }

    except Exception as e:
        print(f"Error in analyze_expenses: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get('/')
def root():
    return RedirectResponse("/api/health")

@app.get('/api/health')
def health_check():
    return {"status": "healthy", "service": "FastAPI AI Service"}

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get('PORT', 10000))
    uvicorn.run("app:app", host='0.0.0.0', port=port)
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from dateutil import parser
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from schemas import UserQuery, AnalyzeRequest
from ai_service import SYSTEM_PROMPT, llm_with_tools, llm, search_expenses

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post('/api/ai/chat')
async def chat(query: UserQuery):
    """
    Process user financial query and return AI-generated response.
    """
    try:
        logger.info(f"Chat query from user {query.userId}: {query.userQuery[:100]}")
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User ID: {query.userId}\n\n{query.userQuery}")
        ]
        
        response = llm_with_tools.invoke(messages)
        logger.debug(f"LLM response type: {type(response).__name__}")
        
        if response.tool_calls:
            logger.info(f"Tool calls detected: {[tc['name'] for tc in response.tool_calls]}")
            
            messages.append(response)
            
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'search_expenses':
                    query_text = tool_call['args'].get('query_text', '')
                    user_id_arg = tool_call['args'].get('user_id', query.userId)
                    logger.debug(f"Executing search_expenses: query='{query_text}'")
                    
                    # Call the actual search function - the wrapper gets underlying function via .invoke or direct call
                    # LangChain tool can be invoked manually
                    result = search_expenses.invoke({
                        "query_text": query_text,
                        "user_id": user_id_arg
                    })
                    logger.debug(f"Tool result: {result[:100]}")
                    
                    messages.append(ToolMessage(
                        tool_call_id=tool_call['id'],
                        content=result
                    ))
            
            final_response = llm.invoke(messages)
            logger.info(f"Generated polished response")
            return final_response.content
        
        logger.info("No tools needed, returning direct response")
        return response.content
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process AI chat")

@router.post('/api/analyze')
def analyze_expenses(body: AnalyzeRequest):
    """
    Analyze expense trends and generate insights.
    """
    try:
        expenses = body.expenses
        logger.info(f"Analyzing {len(expenses)} expenses")
        
        if not expenses:
            return {
                "data": [],
                "insight": "No expenses found for the selected period."
            }

        processed_expenses = []
        for exp in expenses:
            try:
                parsed_date = parser.parse(exp['date'])
                expense_copy = exp.copy()
                expense_copy['_parsed_date'] = parsed_date
                processed_expenses.append(expense_copy)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"Skipping expense {exp.get('_id', 'unknown')}: {e}")
                continue

        processed_expenses.sort(key=lambda x: x['_parsed_date'])

        monthly_data = {}
        for exp in processed_expenses:
            d = exp['_parsed_date']
            key = f"{d.year}-{d.month:02d}"
            if key not in monthly_data:
                monthly_data[key] = 0
            monthly_data[key] += exp['amount']

        monthly_list = [{"month": key, "amount": amt} for key, amt in monthly_data.items()]
        monthly_list.sort(key=lambda x: x['month'])

        if len(monthly_list) >= 2:
            prev, curr = monthly_list[-2], monthly_list[-1]
            if prev["amount"] != 0:
                change = ((curr["amount"] - prev["amount"]) / prev["amount"]) * 100
            else:
                change = 0
            trend = "increased" if change > 0 else "decreased" if change < 0 else "remained the same"
            insight = f"Your expenses have {trend} by {abs(change):.2f}% compared to last month."
        elif len(monthly_list) == 1:
            insight = "This is your first month of tracking expenses. Keep it up!"
        else:
            insight = "Not enough data to generate insights."
        
        logger.info(f"Analysis complete: {len(monthly_list)} months of data analyzed")

        return {
            "data": monthly_list,
            "insight": insight
        }

    except Exception as e:
        logger.error(f"Error in analyze_expenses: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get('/')
def root():
    return RedirectResponse("/api/health")

@router.get('/api/health')
def health_check():
    logger.info("Health check requested")
    return {"status": "healthy", "service": "FastAPI AI Service"}

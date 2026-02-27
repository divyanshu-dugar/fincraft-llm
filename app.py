import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser
from typing import List, Dict, Any
from openai import OpenAI

load_dotenv()

app = FastAPI()

openai_client = OpenAI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    userQuery: str

@app.post('/api/ai/chat')
async def chat(query: UserQuery):
    SYSTEM_PROMPT = """
    You are Fincraft AI - a personal finance mentor. You should only and only answer questions related to personal finance which includes income, saving goals, expense and budget, and excludes anything related to investments. If user asks something other than money management, politely refuse to answer and redirect them to ask something related to personal finance.
    """

    response = openai_client.responses.create(
        model = "gpt-5.2",
        input = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": query.userQuery
            }
        ],
    )

    print(f"🤖: {response.output_text}")
    return response.output_text


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
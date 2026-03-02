from pydantic import BaseModel, Field
from typing import List, Dict, Any

class UserQuery(BaseModel):
    """
    Request model for user financial queries.
    """
    userQuery: str = Field(..., description="User's financial question")
    userId: str = Field(..., description="User's unique ID")

class AnalyzeRequest(BaseModel):
    """
    Request model for expense trend analysis.
    """
    expenses: List[Dict[str, Any]] = Field(..., description="List of expenses to analyze")

from pydantic import BaseModel, Field
from typing import List, Optional


class SinglePredictRequest(BaseModel):
    followers_count: int = Field(..., ge=0, description="Number of followers")
    friends_count: int = Field(..., ge=0, description="Number of friends")
    post_count: int = Field(..., ge=0, description="Number of posts")
    location: str = Field("", description="User location")
    lang: str = Field("en", description="User language code")
    description: str = Field("", description="User profile description")
    created_at: str = Field(..., description="Account creation date (YYYY-MM-DD)")


class SinglePredictResponse(BaseModel):
    prediction: str = Field(..., description="Fake or Real")
    fake_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being fake")
    risk_level: str = Field(..., description="High / Medium / Low")


class BatchPredictItem(BaseModel):
    prediction: str
    fake_probability: float
    risk_level: str


class BatchPredictResponse(BaseModel):
    total: int
    fake_count: int
    high_risk_count: int
    results: List[BatchPredictItem]


class ErrorResponse(BaseModel):
    detail: str

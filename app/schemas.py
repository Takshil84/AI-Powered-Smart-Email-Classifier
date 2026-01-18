from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class EmailRequest(BaseModel):
    content: str
    subject: Optional[str] = None
    sender: Optional[str] = None

class ClassificationResponse(BaseModel):
    category: str
    urgency: str
    confidence: float
    source: str
    source: str
    xai_highlights: Optional[List[Dict[str, Any]]] = []
    alert_flags: Optional[List[str]] = []
    details: Optional[Dict[str, Any]] = None

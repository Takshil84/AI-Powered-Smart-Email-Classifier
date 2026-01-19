from fastapi import APIRouter, HTTPException
from ..schemas import EmailRequest, ClassificationResponse
import sys
import os

# Add parent directory to path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Use rule-based urgency classifier instead of hybrid ML model
    from ..rule_based_urgency import RuleBasedUrgencyClassifier
    from ..category_inference import CategoryClassifier  # <-- FIXED
    
    # Initialize Classifiers (Global instance to avoid reloading)
    urgency_classifier = RuleBasedUrgencyClassifier()
    category_classifier = CategoryClassifier()
    
except ImportError as e:
    urgency_classifier = None
    category_classifier = None
    print(f"Warning: Could not import classifiers: {e}")

router = APIRouter()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_email(email: EmailRequest):
    if not urgency_classifier or not category_classifier:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    combined_text = f"{email.subject} {email.content}" if email.subject else email.content
    
    urgency_result = urgency_classifier.predict(combined_text)
    category_result = category_classifier.predict(combined_text)
    
    combined_details = {
        "urgency_details": urgency_result.get("details"),
        "category_source": category_result.get("source")
    }
    
    alert_flags = []
    if urgency_result["confidence"] < 0.60:
        alert_flags.append("LOW_CONFIDENCE")
    if urgency_result["final_label"] == "High":
        alert_flags.append("CRITICAL_URGENCY")
    
    return ClassificationResponse(
        category=category_result["label"],
        urgency=urgency_result["final_label"],
        confidence=urgency_result["confidence"], 
        source=f"Urgency: {urgency_result['source']} | Category: {category_result['source']}",
        xai_highlights=urgency_result.get("highlights", []),
        alert_flags=alert_flags,
        details=combined_details
    )

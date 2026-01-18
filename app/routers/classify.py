from fastapi import APIRouter, HTTPException
from ..schemas import EmailRequest, ClassificationResponse
import sys
import os

# Add parent directory to path to import local modules (hybrid_inference, category_inference)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from hybrid_inference import HybridUrgencyClassifier
    from app.category_inference import CategoryClassifier
    
    # Initialize Classifiers (Global instance to avoid reloading)
    urgency_classifier = HybridUrgencyClassifier()
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
    
    # Combine Subject and Content for better context
    combined_text = f"{email.subject} {email.content}" if email.subject else email.content
    
    # Urgency Prediction
    urgency_result = urgency_classifier.predict(combined_text)
    
    # Category Prediction
    category_result = category_classifier.predict(combined_text)
    
    # Combine Details
    combined_details = {
        "urgency_details": urgency_result.get("details"),
        "category_source": category_result.get("source")
    }
    
    # Alert Logic
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

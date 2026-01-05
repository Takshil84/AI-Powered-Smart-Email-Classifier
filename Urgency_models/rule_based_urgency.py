import re

# Defined Keyword Lists
# Critical = High Urgency (Immediate Action Required)
CRITICAL_KEYWORDS = [
    r"\bsystem\s+(?:is\s+)?down\b",
    r"\bcrash\b",
    r"\bsecurity breach\b",
    r"\bunable to login\b",
    r"\bdata loss\b",
    r"\bhacked\b",
    r"\bimmediate assistance\b",
    r"\bblocked\b",
    r"\bcritical failure\b",
    r"\bpayment failed\b",
    r"\bserver\s+(?:is\s+)?down\b"
]

# Medium = Medium Urgency (Needs Attention soon)
MEDIUM_KEYWORDS = [
    r"\bhelp\b",
    r"\bissue\b",
    r"\berror\b",
    r"\bbug\b",
    r"\bfail\b",
    r"\bnot working\b",
    r"\brequest\b",
    r"\bstatus\b",
    r"\bupdate\b",
    r"\bcheck\b",
    r"\bunable to\b"
]

def detect_urgency_rules(text):
    """
    Analyzes text for urgency keywords.
    
    Returns:
        dict: {
            "label": "High" | "Medium" | "Low",
            "score": 0, 1, or 2 (mapped to Low, Medium, High),
            "reason": matched_keyword (str) or None,
            "rule_confidence": 1.0 (if match found) else 0.0
        }
    """
    text_lower = text.lower()
    
    # 1. Check for Critical Keywords (High Urgency)
    for pattern in CRITICAL_KEYWORDS:
        if re.search(pattern, text_lower):
            return {
                "label": "High",
                "score": 2,
                "reason": pattern.replace(r"\b", "").replace("\\", ""),
                "rule_confidence": 1.0
            }
            
    # 2. Check for Medium Keywords
    for pattern in MEDIUM_KEYWORDS:
        if re.search(pattern, text_lower):
            return {
                "label": "Medium",
                "score": 1,
                "reason": pattern.replace(r"\b", "").replace("\\", ""),
                "rule_confidence": 0.8  # Medium rules are less absolute than critical ones
            }
            
    # 3. Default to Low if no keywords found
    return {
        "label": "Low",
        "score": 0,
        "reason": None,
        "rule_confidence": 0.0
    }

# Quick Test
if __name__ == "__main__":
    test_cases = [
        "The system is down and I cannot access the portal.",
        "I need help with my refund status.",
        "Just wanted to say thanks for the great service.",
        "There is a critical failure in the payments module."
    ]
    
    print("--- Rule Based Verification ---")
    for t in test_cases:
        print(f"Text: '{t}'\nResult: {detect_urgency_rules(t)}\n")

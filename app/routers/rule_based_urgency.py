import re

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
    text_lower = text.lower()

    for pattern in CRITICAL_KEYWORDS:
        if re.search(pattern, text_lower):
            return {
                "label": "High",
                "score": 2,
                "reason": re.sub(r"\\b", "", pattern),
                "rule_confidence": 1.0
            }

    for pattern in MEDIUM_KEYWORDS:
        if re.search(pattern, text_lower):
            return {
                "label": "Medium",
                "score": 1,
                "reason": re.sub(r"\\b", "", pattern),
                "rule_confidence": 0.8
            }

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

    for t in test_cases:
        print(f"{t}\n=> {detect_urgency_rules(t)}\n")

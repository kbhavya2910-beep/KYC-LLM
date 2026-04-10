import sys, json
sys.path.insert(0, 'backend')
from llm_decision import get_llm_decision

tests = [
    ("GENUINE - good match, liveness=0",   85,  0, False, "center", 10),
    ("GENUINE - good match, liveness=50",  82, 50, False, "center", 10),
    ("GENUINE - full pass",                88,100, True,  "left",   10),
    ("SUSPICIOUS - moderate",              65, 50, False, "center", 10),
    ("FRAUDULENT - low face match",        40, 30, False, "center", 10),
    ("SUSPICIOUS - no_face but good match",80,  0, False, "no_face",10),
]

for label, fm, ls, blink, head, df in tests:
    r = get_llm_decision(fm, ls, blink, head, df)
    print(f"\n[{label}]")
    print(f"  risk={r['risk_score']}")
    print(f"  reasoning: {r['reasoning']}")

# utils/verify_setup.py
import sys

checks = {}

try:
    import torch; checks["torch"] = torch.__version__
except: checks["torch"] = "FAILED"

try:
    import langgraph; checks["langgraph"] = "OK"
except: checks["langgraph"] = "FAILED"

try:
    import openai; checks["openai"] = openai.__version__
except: checks["openai"] = "FAILED"

try:
    from sentence_transformers import SentenceTransformer
    checks["sentence-transformers (mCLIP)"] = "OK"
except: checks["sentence-transformers (mCLIP)"] = "FAILED"

try:
    import easyocr; checks["easyocr"] = "OK"
except: checks["easyocr"] = "FAILED"

try:
    from deep_translator import GoogleTranslator
    checks["deep-translator"] = "OK"
except: checks["deep-translator"] = "FAILED"

try:
    from vendi_score import vendi; checks["vendi-score"] = "OK"
except: checks["vendi-score"] = "FAILED"

try:
    import streamlit; checks["streamlit"] = streamlit.__version__
except: checks["streamlit"] = "FAILED"

try:
    import krippendorff; checks["krippendorff"] = "OK"
except: checks["krippendorff"] = "FAILED"

try:
    import Levenshtein; checks["Levenshtein"] = "OK"
except: checks["Levenshtein"] = "FAILED"

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
        "DEEPL_AUTH_KEY": os.getenv("DEEPL_AUTH_KEY"),
    }
    for k, v in keys.items():
        checks[k] = "✅ SET" if v and v != "your_key_here" and v != "not_used" else "❌ MISSING"
except: checks[".env loading"] = "FAILED"

print("\n── MulBrandEval Setup Verification ──")
all_ok = True
for pkg, status in checks.items():
    icon = "✅" if "FAILED" not in str(status) and "MISSING" not in str(status) else "❌"
    print(f"  {icon}  {pkg:<35} {status}")
    if "FAILED" in str(status) or "MISSING" in str(status):
        all_ok = False

print("\n" + ("✅ All systems go. Ready for Phase 1A." if all_ok else
              "❌ Fix the items above before proceeding."))
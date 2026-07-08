from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv(dotenv_path=Path(__file__).parent / ".env", override=True)
key = os.environ.get("GOOGLE_API_KEY", "")
print("Key suffix:", "..."+key[-6:] if key else "EMPTY")
import google.generativeai as genai
genai.configure(api_key=key)
try:
    models = list(genai.list_models())
    print("OK - models count:", len(models))
except Exception as e:
    print("ERROR:", e)

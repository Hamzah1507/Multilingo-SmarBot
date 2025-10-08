import os
import json
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from deep_translator import GoogleTranslator
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv()
# Read API key from env
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    api_key=API_KEY  # explicitly pass the API key
)

# Initialize Flask
app = Flask(__name__)

# Load knowledge base
try:
    with open("knowledge_base.json", "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)
except FileNotFoundError:
    print("Warning: knowledge_base.json not found. Using empty knowledge base.")
    knowledge_base = {}

# Response cache for English & translations
response_cache = {}

# Supported languages
LANG_MAP = {"en": "en", "hi": "hi", "gu": "gu", "ta": "ta", "mr": "mr"}

# ------------------- Helper function -------------------
def generate_response_with_gemini(user_query, language="en"):
    cache_key = f"{user_query}_{language}"
    if cache_key in response_cache:
        return response_cache[cache_key]

    # English response first
    english_key = f"{user_query}_en"
    if english_key in response_cache:
        english_response = response_cache[english_key]
    else:
        prompt = f"""
You are a helpful University support chatbot. 
Answer ONLY using the knowledge base. If the answer is not found, politely say you don't know.

Knowledge Base:
{json.dumps(knowledge_base, indent=2)}

User Query: {user_query}
"""
        try:
            english_response = llm.invoke(prompt).content
        except ResourceExhausted as e:
            print(f"Gemini quota exceeded! Details: {e}")
            english_response = "Sorry, the API quota is exceeded. Try again later."
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            english_response = "Sorry, I'm having trouble connecting right now."

        response_cache[english_key] = english_response

    # Return English immediately if requested
    if language == "en":
        response_cache[cache_key] = english_response
        return english_response

    # Translate if needed
    target_lang = LANG_MAP.get(language, "en")
    translation_key = f"{user_query}_{target_lang}"
    if translation_key in response_cache:
        translated_response = response_cache[translation_key]
    else:
        try:
            translated_response = GoogleTranslator(source="auto", target=target_lang).translate(english_response)
        except Exception as e:
            print(f"Translation failed for {target_lang}: {e}")
            translated_response = english_response
        response_cache[translation_key] = translated_response

    response_cache[cache_key] = translated_response
    return translated_response

# ------------------- Routes -------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.form.get("user_query")
    language = request.form.get("language", "en")

    if not user_query:
        return jsonify({"response": "Please enter a question."})

    response_text = generate_response_with_gemini(user_query, language)
    return jsonify({"response": response_text})

@app.route("/test")
def test():
    return "Flask server is working!"

# ------------------- Run server -------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

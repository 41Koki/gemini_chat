import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Set the API key for Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to generate a response from the model
gemini = genai.GenerativeModel("gemini-1.5-flash")
prompt = "こんにちは"
response = gemini.generate_content(prompt)
print(response.text)


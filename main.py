from google.genai import models
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = api_key
)

response = llm.invoke("Explain machine learning in one sentence.")

print(response.content)

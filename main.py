
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

with open("knowledge.txt", "r", encoding="utf-8") as file:
    document = file.read()

prompt = ChatPromptTemplate.from_template(
    """
    You are answering questions about the provided document.

    DOCUMENT:
    {document}

    QUESTION:
    {question}

    Answer the question using only the information provided in the document.
    If the answer is not present in the document, say that the document does not contain the answer.
    """
)


llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = api_key
)



chain = prompt | llm

response = chain.invoke({
    "document": document,
    "question": "Who invented the Python programming language?"
})

print(response.content)





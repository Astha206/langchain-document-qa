
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

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


class QAResponse(BaseModel):
    answer: str
    source_found: bool

llm = ChatGoogleGenerativeAI(
    model = "gemini-3.6-flash",
    google_api_key = api_key
)

structured_llm = llm.with_structured_output(QAResponse)


chain = prompt | structured_llm

response = chain.invoke({
    "document": document,
    "question": "Who invented the Python programming language?"
})

print(response.answer)
print(response.source_found)







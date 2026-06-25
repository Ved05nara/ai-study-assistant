import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)


def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an expert college study assistant.

    Rules:
    1. Answer only from the provided context.
    2. Explain concepts in simple student-friendly language.
    3. If possible, give examples.
    4. Structure answers using bullet points.
    5. If the answer is not present in the context, say:
    "I could not find that information in the uploaded notes."

    Context:
    {context}

    Question:
    {question}
    """

    response = model.generate_content(prompt)

    return response.text
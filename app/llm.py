
from langchain_core.messages import AIMessage, HumanMessage

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))



def generate_answer(context, query):
    messages = [
        {"role": "system", "content": "Answer only using the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]

    result = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # model goes here
        messages=messages
    )

    return result.choices[0].message.content

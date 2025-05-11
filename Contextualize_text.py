import spacy
from groq import Groq
import dotenv
import os
from dotenv import load_dotenv

load_dotenv()

# Use your API key securely (e.g., with environment variables in production)
client = Groq(api_key = os.getenv("GROQ_API_KEY"))

nlp = spacy.load("en_core_web_sm")

def extract_numerical_context(text):
    doc = nlp(text)
    numerical_contexts = []
    for token in doc:
        print(token.text, token.pos_, token.dep_)
        if token.like_num: 
            start_index = max(0, token.i - 3)  # Get words 3 before
            end_index = min(len(doc), token.i + 3)  # Get words 3 after
            context = " ".join(
                [doc[i].text for i in range(start_index, end_index)]
            )
            numerical_contexts.append((token.text, context))
    return numerical_contexts

def define_numericals(num, context):
    prompt = f"Based on the context '{context}', what can you infer about the '{num}'? For example, is it a date, a price, a quantity, etc.? Give an explaination."
    
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a linguist and a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    answer = response.choices[0].message.content
    return answer

# Example usage
text = "The price is $100, and I bought 3 items on 2024-05-08.  The car's top speed is 200 km/h."
answer_texts = []
contexts = extract_numerical_context(text)
for num, context in contexts:
    answer = define_numericals(num, context)
    answer_text = f"Number: {num}, Context: {context}, Answer: {answer}"
    try:
        with open("answer.txt", "a") as f:
            f.write(f"{answer_text}\n")  # Write each item on a new line
        print(f"Numerical contexts saved to answer.txt.")
    except Exception as e:
        print(f"Error saving to file: {e}")

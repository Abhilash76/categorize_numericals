import spacy
from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_numerical_context(text: str, nlp_model = nlp) -> list:
    """Extracts numerical values and their surrounding context from the given text."""
    doc = nlp_model(text)
    numerical_contexts = []
    for token in doc:
        if token.like_num:
            start_index = max(0, token.i - 5)
            end_index = min(len(doc), token.i + 5)
            context = " ".join(
                [doc[i].text for i in range(start_index, end_index)]
            )
            numerical_contexts.append((token.text, context))
    return numerical_contexts


def generate_text_for_speech(number: str, context: str, groq_client = client) -> str:
    """Generates a natural language representation of the number based on its context."""
    prompt = f"""
    Given the number "{number}" and its surrounding context "{context}", generate a natural language
    representation that is suitable for speech.  Consider whether the number is a date, time,
    quantity, measurement, or other type of value.  Provide only the textual representation.

    Examples:
    - Input: number="01.05", context="International Labor Day is celebrated on 01.05 every year"
      Output: ""International Labor Day is celebrated on First of May every year"
    - Input: number="2023-11-15", context="The meeting is scheduled for 2023-11-15"
      Output: "The meeting is scheduled for Fifteenth of November, two thousand twenty-three"
    - Input: number="12", context="Please attend on 12.12"
      Output: "Please attend on Twelfth of December"
    - Input: number="30C", context="The temperature is around 30C"
      Output: "The temperature is around Thirty degrees Celsius"
    - Input: number="3", context="bought 3 items"
      Output: "bought three items"
    - Input: number = "100", context = "price is $100"
      Output: "price is one hundred dollars"

    Do not include any extra words or punctuation.
    """
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a linguist and a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content


def define_numericals(num: str, context: str, groq_client: Groq = None) -> str:
    """Defines the numerical value in a natural language format based on its context."""
    if groq_client is None:
        groq_client = client

    speech_text = generate_text_for_speech(num, context, groq_client)

    return f"Number: {num}, Context: {context}, Answer: {speech_text}"


def process_and_save_numericals(text: str, filename: str = "num_to_text.txt",
                               nlp_model = nlp,
                               groq_client = client) -> None:
    """Processes the text to extract numerical values and their contexts, then saves the results."""
    contexts = extract_numerical_context(text, nlp_model)
    if not contexts:
        print("No numerical values found in the text.")
        return

    try:
        with open(filename, "a") as f:
            for num, context in contexts:
                answer_text = define_numericals(num, context, groq_client)
                f.write(f"{answer_text}\n")
            print(f"Numerical contexts saved to {filename}.")
            f.close()
            with open(filename, "r") as f:
                context = f.read()
                while len(context) == 0:
                    if len(context) > 0:
                        break
                f.close()

                prompt = f"""Modify the {text} by converting all the numericals to textual format with the given context {context} and generate a natural language representation that is suitable for speech. Do not generate any greeatings and once the modified text is generated, please stop."""
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": "You are a linguist and a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                )
                return response.choices[0].message.content
    except Exception as e:
        print(f"Error saving to file: {e}")


if __name__ == "__main__":
    text = "It's a sunny day outside. The temperature is around 30C. The price is $100, and I bought 3 items on 10.05.2025. The car's top speed is 200 km/h."
    final_response = process_and_save_numericals(text)
    with open("answer.txt", "w") as f:
        f.write(final_response)
        f.close()
        print(f"Final response saved to answer.txt.")

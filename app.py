import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch


# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")


with open("assist_ai_database1.txt", "r", encoding="utf-8") as file:
    assist_ai_database1_text = file.read()
print(assist_ai_database1_text)

def preprocess_text(text):
 
  cleaned_text = text.strip()

  chunks = cleaned_text.split("\n")


  cleaned_chunks = []

  for chunk in chunks:
    stripped_chunk = chunk.strip()
      
    if len(stripped_chunk) > 0:
      cleaned_chunks.append(stripped_chunk)

return cleaned_chunks

cleaned_chunks = preprocess_text(assist_ai_database1_text)

print(cleaned_chunks)


def respond(message, history):
    messages = [{"role": "system", "content": "You are a chatbot designed to help students create a study schedule to reduce the amount of stress they expereince during the school year. Firstly, ask the students what subjects they take, their current grade in each, then the grade that they want to acheive. Please ask the student when they next assignment/exam/test is for their chosen subjects. Then ask what commitments do they have outside of school e.g. clubs, chores ect."}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=100,
        temperature=1.3
    )

    return response.choices[0].message.content.strip()

with gr.Blocks() as dashboard:

    gr.Markdown("# Good morning, Scholar!")
    gr.Markdown("Let's make today productive.")

    with gr.Row():
        with gr.Column():
            chatbot = gr.ChatInterface(respond)

dashboard.launch()


# TODO: This is just a starting point! Customize the system prompt,!
# the model, and the interface to make this project your own!

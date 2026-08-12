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

with open("calendar.html", "r", encoding="utf-8") as file:
        calendar_html = file.read()

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

# Load the pre-trained embedding model that converts text to vectors

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_embeddings(text_chunks):
    chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)

    return chunk_embeddings


chunk_embeddings = create_embeddings(cleaned_chunks)

print(chunk_embeddings.shape)


def get_top_chunks(query, chunk_embeddings, text_chunks):
  
  query_embedding = model.encode(query, convert_to_tensor=True)

  query_embedding_normalized = query_embedding / query_embedding.norm()

  chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)

  similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) 

  print(similarities)

  top_indices = torch.topk(similarities, k=3).indices

  top_chunks = []

  for i in top_indices:
    chunk = text_chunks[i]
    top_chunks.append(chunk)

  return top_chunks



def respond(message, history):

    top_results = get_top_chunks(
        message,
        chunk_embeddings,
        cleaned_chunks
    )

    study_context = "\n".join(top_results)

    messages = [{"role": "system", "content": f""" You are a friendly study-planning assistant designed to help students create realistic, personalised study schedules while reducing stress.

First, collect the information you need from the student. Ask about:
- The subjects they study
- Their current grade in each subject
- Their target grade in each subject
- The dates of upcoming exams, tests, assignments, or deadlines
- Their regular commitments, such as school, clubs, sports, work, chores, or appointments
- When they are normally available to study

Do not ask all of these questions at once. Have a natural conversation with the student and ask short, clear questions.

Once you have enough information, create a personalised study schedule.

When creating the schedule:
- Prioritise subjects with upcoming deadlines
- Give more attention to subjects where the student is further from their target grade
- Do not schedule study sessions during the student's existing commitments
- Include regular breaks
- Avoid overwhelming the student with too many study sessions
- Give specific study times and subjects
- Make the schedule clear and easy to follow
- Be sure to use relevant characters fetched from the database, if user query is in english, maintain usage of english language. 

Use the following relevant information retrieved from the study database: {study_context} Use the database information as guidance when creating your response. Do not mention the database, RAG, embeddings, or retrieved chunks to the student. If you do not yet have enough information to create a useful schedule, continue asking the student for the missing information instead of making up details.
"""}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=100,
        temperature=1.3
    )

    return response.choices[0].message.content.strip()

# Adds a new task to the task list
def add_new_task(task, current_tasks):
    if not task or task.strip() == "":
        return current_tasks, ""

    current_tasks.append({"text": task.strip(), "completed": False})
    return current_tasks, ""

# Removes a task when its checkbox is checked
def update_task(task_text, completed, current_tasks):
    if completed:
        new_tasks = []

        for task in current_tasks:
            if task["text"] != task_text:
                new_tasks.append(task)
        return new_tasks

    return current_tasks

with gr.Blocks() as dashboard:

    gr.Markdown("# Good morning, Scholar!")
    gr.Markdown("Let's make today productive.")

    with gr.Row():
        with gr.Column():
            chatbot = gr.ChatInterface(respond)

        with gr.Column():
            gr.Markdown("## Today's Tasks")

            with gr.Group():

                # Stores all tasks
                tasks = gr.State([])
    
                # Display tasks
                @gr.render(inputs=tasks)
                def render_tasks(current_tasks):
    
                    for task in current_tasks:
                        checkbox = gr.Checkbox(
                            label = task["text"],
                            value=task["completed"]
                        )
                        # Removes the task when it is checked
                        checkbox.change(
                            update_task,
                            inputs=[
                                gr.State(task["text"]),
                                checkbox,
                                tasks
                            ],
                            outputs=tasks
                        )
                
                #Input for a new task
                with gr.Row(elem_id="task-input-row"):
                    task_input = gr.Textbox(show_label = False, placeholder="Add a task...", scale = 4, elem_id = "task-input")
                    add_task = gr.Button("+", scale = 0, elem_id = "add-task-button")
                
                # Add task when button is clicked
                add_task.click(
                    add_new_task,
                    inputs=[task_input, tasks],
                    outputs=[tasks, task_input]
                )

                # Add task when Enter is pressed
                task_input.submit(
                    add_new_task,
                    inputs=[task_input, tasks],
                    outputs=[tasks, task_input]
                )
            
            gr.Markdown("## Pomodoro Timer")
            # We can change the design of the timer. This is just here temporarily.
            gr.HTML("""<iframe src="https://widgets.commoninja.com/iframe/8318698e-d3c1-4004-b137-3d3e750b45ee" width="75%" height="500" frameborder="0" scrolling="no"></iframe>""")
            
        with gr.Column():
            gr.Markdown("### Calendar View Mode")
            # 2. Pass the file contents directly into the HTML component
            gr.HTML(calendar_html)

dashboard.launch(theme=gr.themes.Soft())


# TODO: This is just a starting point! Customize the system prompt,!
# the model, and the interface to make this project your own!

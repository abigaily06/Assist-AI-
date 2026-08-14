import gradio as gr
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import torch

# Initialize the Hugging Face Inference Client
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")

# 1. Load your external files safely
try:
    with open("calendar.html", "r", encoding="utf-8") as file:
        calendar_html = file.read()
except FileNotFoundError:
    calendar_html = "<p>Calendar file not found.</p>"

try:
    with open("assist_ai_database1.txt", "r", encoding="utf-8") as file:
        assist_ai_database1_text = file.read()
except FileNotFoundError:
    assist_ai_database1_text = ""

try:
    with open("style.css", "r", encoding="utf-8") as file:
        custom_css = file.read()
except FileNotFoundError:
    custom_css = ""

# 2. Text Preprocessing and RAG setup
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

# Load the pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(text_chunks):
    chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return chunk_embeddings

if cleaned_chunks:
    chunk_embeddings = create_embeddings(cleaned_chunks)
else:
    chunk_embeddings = torch.empty((0, 384)) # Fallback if text is empty

def get_top_chunks(query, chunk_embeddings, text_chunks):
    if len(text_chunks) == 0:
        return []
        
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
    
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized) 

    k_val = min(3, len(text_chunks))
    top_indices = torch.topk(similarities, k=k_val).indices
    
    top_chunks = []
    for i in top_indices:
        chunk = text_chunks[i]
        top_chunks.append(chunk)
    return top_chunks

def respond(message, history, current_tasks):
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

When a user mentions a new assignment, you must add it to their list by typing the secret code [ADD_TASK] followed exactly by the task name. For example: 'I can help with that! [ADD_TASK] Finish Math Worksheet'. Do not add anything after the task name. Extract only the name of task. Do not use the secret code unless you are adding a task.

IMPORTANT:
Do not write anything after the task name.
The task name must contain ONLY the assignment name.
Do not include explanations, punctuation, or additional text after the task name.
Do not use [ADD_TASK] unless you are adding a task.

TASK ADDING RULE:
You must follow the [ADD_TASK] rule EVERY TIME the user asks you to add an assignment, even if you have already added an assignment earlier in the conversation.

Do not skip the [ADD_TASK] code for later assignments.

For example:
User: Add my math homework.
Assistant: [ADD_TASK] Math Homework

User: Also add my biology project.
Assistant: [ADD_TASK] Biology Project

User: And add my English essay.
Assistant: [ADD_TASK] English Essay

Use the following relevant information retrieved from the study database: {study_context} Use the database information as guidance when creating your response. Do not mention the database, RAG, embeddings, or retrieved chunks to the student. If you do not yet have enough information to create a useful schedule, continue asking the student for the missing information instead of making up details.
"""}]

    # 1. Extend the messages list with the Gradio history dictionaries
    if history:
        messages.extend(history)
            
    messages.append({"role": "user", "content": message})
    
    response = client.chat_completion(
        messages,
        max_tokens=500,
        temperature=1.3
    )
    
    ai_response = response.choices[0].message.content.strip()
    
    # Process the secret code IF it exists
    if "[ADD_TASK]" in ai_response:
        parts = ai_response.split("[ADD_TASK]")
        
        # 1. Take everything after the secret code and split it by new lines
        after_trigger = parts[1].strip()
        lines = after_trigger.split("\n")
        
        # 2. The task name is ONLY the first line
        task_name = lines[0].strip()
        
        # 3. The rest of the AI's question is everything else combined back together
        rest_of_message = "\n".join(lines[1:]).strip()
        
        # Add the task to the list (using the + method for Gradio state!)
        current_tasks = current_tasks + [{"text": task_name, "completed": False}]
        
        # Format the final visible message to include the prefix, the success text, and the suffix!
        ai_response = parts[0].strip() + f"\n\n*(Added '{task_name}' to your task list!)*\n\n{rest_of_message}".strip()

    # 2. Save the history using ONLY the dictionary format required by Gradio 5+
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ai_response})
    
    return "", history, current_tasks

# Task Management Functions
def add_new_task(task, current_tasks):
    if not task or task.strip() == "":
        return current_tasks, ""
    current_tasks.append({"text": task.strip(), "completed": False})
    return current_tasks, ""

def update_task(task_text, completed, current_tasks):
    if completed:
        new_tasks = []
        for task in current_tasks:
            if task["text"] != task_text:
                new_tasks.append(task)
        return new_tasks
    return current_tasks

# Interface Layout
with gr.Blocks() as dashboard:
    with gr.Row():
        gr.HTML('<img src="file=Untitled_design-removebg-preview.png" style="width: 80px; height: 80px; object-fit: contain;">')
        with gr.Column():
            gr.Markdown("# Good morning, Scholar!")
            gr.Markdown("Let's make today productive.")

    # Global state for tasks
    tasks = gr.State([])

    with gr.Row():

        # COLUMN 1: Chatbot
        with gr.Column():
            gr.Markdown("### Assistant")
            chatbot = gr.Chatbot(avatar_images=(None, "Untitled_design-removebg-preview.png"))
            

            # Exemplar prompt buttons
            with gr.Row():
                schedule_prompt = gr.Button("Make a schedule")
                homework_prompt = gr.Button("Homework help")
                exam_prompt = gr.Button("Exam prep")

            # Message box
            msg = gr.Textbox(
                placeholder="Tell me your tasks...",
                show_label=False
            )

            # Clicking a prompt puts it into the message box
            schedule_prompt.click(
                fn=lambda: "Help me create a study schedule",
                outputs=msg
            )

            homework_prompt.click(
                fn=lambda: "Help me organise my homework",
                outputs=msg
            )

            exam_prompt.click(
                fn=lambda: "Help me prepare for an upcoming exam",
                outputs=msg
            )

            # Press Enter to send message to AI
            msg.submit(
                fn=respond,
                inputs=[msg, chatbot, tasks],
                outputs=[msg, chatbot, tasks]
            )


        # COLUMN 2: Tasks & Timer
        with gr.Column():
            gr.Markdown("## Today's Tasks")

            with gr.Group():

                # Dynamically display tasks
                @gr.render(inputs=tasks)
                def render_tasks(current_tasks):
                    for task in current_tasks:
                        checkbox = gr.Checkbox(
                            label=task["text"],
                            value=task["completed"]
                        )

                        checkbox.change(
                            update_task,
                            inputs=[
                                gr.State(task["text"]),
                                checkbox,
                                tasks
                            ],
                            outputs=tasks
                        )

                # Task input
                with gr.Row(elem_id="task-input-row"):
                    task_input = gr.Textbox(
                        show_label=False,
                        placeholder="Add a task...",
                        scale=4,
                        lines=1,
                        elem_id="task-input"
                    )

                    add_task = gr.Button(
                        "+",
                        scale=0,
                        elem_id="add-task-button"
                    )

                # Add task when + is clicked
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

            # Pomodoro Timer
            gr.Markdown("## Pomodoro Timer")

            gr.HTML(
                """
                <iframe
                    src="https://widgets.commoninja.com/iframe/d7099aad-af9b-4a30-be44-08e097b41209"
                    width="200%"
                    height="700"
                    frameborder="0"
                    scrolling="no"
                    style="transform: translateX(-50px) scale(0.6);
                           transform-origin: top left;">
                </iframe>
                """
            )


        # COLUMN 3: Calendar
        with gr.Column():
            gr.Markdown("## Calendar View Mode")
            gr.HTML(calendar_html, elem_id="calendar-container")


# Launch dashboard
dashboard.launch(
    theme=gr.Theme.from_hub("hmb/vaporwave"),
    css=custom_css
)
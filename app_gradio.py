import os
import faiss
import numpy as np
import google.generativeai as genai
import PyPDF2
import gradio as gr
from dotenv import load_dotenv

# Load API key from environment
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Set your key in .env or directly


# Global variables to retain state between inputs
index = None
chunk_lookup = {}
embedding_model = "models/embedding-001"
chat_model = genai.GenerativeModel("gemini-1.5-flash")

# --- Utility Functions ---

def read_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        return None
    return text

def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def process_pdf(pdf):
    global index, chunk_lookup
    text = read_pdf(pdf.name)
    if not text:
        return "❌ Failed to read PDF. Try another file."
    chunks = chunk_text(text)
    embeddings = []
    chunk_lookup = {}

    for i, chunk in enumerate(chunks):
        response = genai.embed_content(
            model=embedding_model,
            content=chunk,
            task_type="RETRIEVAL_DOCUMENT"
        )
        embeddings.append(response["embedding"])
        chunk_lookup[i] = chunk

    embedding_matrix = np.array(embeddings).astype("float32")
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return f"✅ PDF processed. {len(chunks)} text chunks embedded."

def ask_question(user_message, history):
    global index, chunk_lookup
    if index is None:
        return history + [[user_message, "⚠️ Please upload and process a PDF first."]]

    # Embed user query
    query_embedding = genai.embed_content(
        model=embedding_model,
        content=user_message,
        task_type="RETRIEVAL_QUERY"
    )["embedding"]
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)

    # FAISS search
    D, I = index.search(query_vector, k=3)
    context_chunks = [chunk_lookup[i] for i in I[0] if i in chunk_lookup]
    if not context_chunks:
        return history + [[user_message, "🤖 I couldn't find relevant info in the report."]]

    # Create prompt with context
    context = "\n".join(context_chunks)
    prompt = f"""
You are a helpful medical assistant. Based on the document excerpts below, answer accurately.

Document Context:
{context}

User Question:
{user_message}

Instructions:
- If unsure, say "I'm sorry, I couldn't find that in the report."
- If medical advice is needed, suggest a professional consultation.
"""

    response = chat_model.generate_content(prompt)
    return history + [[user_message, response.text.strip()]]
with gr.Blocks() as demo:
    gr.Markdown("## 🧬 Chat with Your Lab Report PDF")
    
    with gr.Row():
        file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        process_button = gr.Button("Process PDF")
    
    status = gr.Textbox(label="Status", interactive=False)

    chatbot = gr.Chatbot(label="Chat with PDF", height=400)
    user_input = gr.Textbox(placeholder="Ask something about the report...", label="Your Message")
    send_button = gr.Button("Send")

    clear_button = gr.Button("Clear Chat")

    process_button.click(process_pdf, inputs=file_input, outputs=status)
    send_button.click(fn=ask_question, inputs=[user_input, chatbot], outputs=chatbot)
    clear_button.click(fn=lambda: [], inputs=None, outputs=chatbot)

demo.launch()

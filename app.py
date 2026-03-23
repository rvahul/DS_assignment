import requests
import base64
from collections import deque
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Configuration & Data ---
TELEGRAM_TOKEN = "8608511197:AAEx99nmHH3K1RE59zHgDRsM1TpRUMpFhWM"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Knowledge base of 3-5 documents [cite: 19]
DOCS = [
    "Policy: Remote work is allowed 2 days a week.",
    "Hours: Office is open 9 AM to 6 PM.",
    "Support: Contact IT at ext 404 for password resets.",
    "Kitchen: Label all food in the shared fridge.",
    "Security: Always wear your ID badge in the building."
]

# Session memory (Last 3 interactions per user) 
user_sessions = {}

# Local embedding model [cite: 22, 56]
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(DOCS)

# --- 2. Logic Engines ---

def get_rag_response(user_id, query):
    # Retrieve top relevant chunk [cite: 25]
    query_vec = embed_model.encode([query])
    top_idx = cosine_similarity(query_vec, doc_embeddings).argmax()
    context = DOCS[top_idx]
    
    # Get history 
    history = "\n".join(user_sessions.get(user_id, []))
    
    # Prompt with context + history [cite: 26]
    prompt = f"History:\n{history}\n\nContext: {context}\nQuestion: {query}\nAnswer briefly."
    resp = requests.post(OLLAMA_URL, json={"model": "llama3", "prompt": prompt, "stream": False})
    return resp.json().get("response"), context

async def get_vision_response(image_bytes):
    img_str = base64.b64encode(image_bytes).decode('utf-8')
    # Generate caption + 3 tags [cite: 34, 35]
    prompt = "Describe this image in one sentence and list 3 tags."
    resp = requests.post(OLLAMA_URL, json={
        "model": "llava", 
        "prompt": prompt, 
        "images": [img_str], 
        "stream": False
    })
    return resp.json().get("response")

# --- 3. Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Bot active! Use /ask for questions or upload an image.")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Please use /ask <your question>")
        return
    
    answer, source = get_rag_response(user_id, query)
    
    # Update History 
    if user_id not in user_sessions: user_sessions[user_id] = deque(maxlen=3)
    user_sessions[user_id].append(f"Q: {query} A: {answer}")
    
    await update.message.reply_text(f"{answer}\n\n📌 Source: {source}") # [cite: 41]

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo_file = await update.message.photo[-1].get_file()
    image_bits = await photo_file.download_as_bytearray()
    
    await update.message.reply_text("Processing image... 👁️")
    result = await get_vision_response(image_bits)
    await update.message.reply_text(result)

# --- 4. Run ---
if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask)) # [cite: 13]
    app.add_handler(MessageHandler(filters.PHOTO, image_handler)) # [cite: 14]
    print("Bot is running...")
    app.run_polling()
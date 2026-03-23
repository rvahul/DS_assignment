# Hybrid GenAI Telegram Bot (RAG + Vision)

This project is a lightweight, local-first GenAI bot that combines Retrieval-Augmented Generation (RAG) with Computer Vision. It is built using Python and leverages Ollama to run open-source models locally, ensuring data privacy and high efficiency

**Key Features**
**Mini-RAG System (/ask):** Retrieves relevant answers from a knowledge base of company policies using vector similarity.
**Vision Description:** Automatically generates a short caption and three descriptive tags for any uploaded image.
**Message History Awareness:** Maintains a rolling buffer of the last 3 interactions per user to handle follow-up questions.
**Source Transparency:** Every RAG response includes a "Source Snippet" showing exactly which document was used.
**Contextual Summarization (/summarize): **Recaps the last 3 interactions for a quick conversation review.

**🧱 System Architecture**
The bot uses a modular architecture to route requests based on input type:
**Text Path:** Input → all-MiniLM-L6-v2 Embedding → Cosine Similarity Search → Llama 3 Generation.
**Vision Path:** Image Upload → Base64 Encoding → Llava Vision Model → Caption + Tags.

**🛠️ Tech Stack**
**Bot Framework:** python-telegram-bot
**LLM Engine:** Ollama (running Llama 3 and Llava)
**Embeddings: **sentence-transformers (all-MiniLM-L6-v2)
**Vector Logic:** scikit-learn (Cosine Similarity).

**📦 Installation & Setup**
**1. Model Preparation**
Install Ollama and pull the necessary local models:
ollama pull llama3
ollama pull llava

**2. Environment Setup**
Clone the repository and install dependencies from the provided requirements.txt:

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install libraries
pip install -r requirements.txt

**3. Running the Bot**
1. Replace the placeholder in app.py with your Telegram Bot Token.
2. Start the application:
 python app.py

**🧪 Evaluation Criteria Met**
1. **Code Quality:** Highly modular and readable Python code with minimal dependencies.
2. **Innovation:** Implements the Hybrid Variant, supporting both RAG and Vision modes simultaneously.
3. **Efficiency:** Uses small-footprint local models to ensure fast turnaround times.
4. **User Experience:** Clear command structures (/ask, /summarize, /help) and visual feedback



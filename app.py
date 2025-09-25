# app.py

import requests
import streamlit as st
import numpy as np
import pandas as pd

import pdfplumber
import re
from openai import OpenAI
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import faiss
from ipywidgets import interact_manual, Text, Output
import IPython.display as display

# -------------------------------
# 1. Helper functions
# -------------------------------

secret_value_1 = st.secrets["nvapi_key"]
secret_value_0 = st.secrets["api_key"]

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=secret_value_1
)

# Chunking We using the Jina AI

secret_value_0 = st.secrets["api_key"]

# Importing PDF Using pdfplumber

with pdfplumber.open(r"C:\\Users\\SHYAM SASHANK\\OneDrive\\Desktop\\Insurance_ChatBot\\ICICI_Insurance.pdf") as pdf:
    all_text=''
    for page in pdf.pages:
        all_text += page.extract_text()

print(all_text[:100])
    
# 1. Cleaning of The PDF

def clean_pdf(text):

    text = re.sub(r'Page \d+\n', '', text)  # Remove page numbers
    text = re.sub(r'ICICI .*?\n', '', text) # Remove ICICI header/footer
    text = re.sub(r'IRDAI Regn.*?\n', '', text) # Remove regulator lines
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text) # Fix hyphenated words
    text = re.sub(r'[\u2022•]', '-', text) # Replace bullets with dash
    text = re.sub(r'(?<![.!?])\n', ' ', text) # Join lines that don't end with punctuation
    text = re.sub(r'\n+', '\n', text) # Remove extra newlines
    text = re.sub(r' {2,}', ' ', text) # Remove extra spaces
    return(text)

# Implementing the Function on the All_text

pdf_cleaned = clean_pdf(all_text)
print(pdf_cleaned[:1000])

# Importing the ICICI Website

def read_website(url, secret_value_0):
    # Jina Reader expects the format: https://r.jina.ai/<full-url>
    api_url = f"https://r.jina.ai/{url}"
    headers = {"Authorization": f"Bearer {secret_value_0}"}
    response = requests.get(api_url, headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to read website: {response.status_code}, {response.text}")

url = "https://www.icicibank.com/personal-banking/insurance?ITM=nli_home_na_megamenuItem_1CTA_CMS_insurance_viewAllInsurance_NLI"
website_text = read_website(url, secret_value_0)

print(website_text[:500])  # Just show first 1000 chars
combined_text = (
    "STATIC PDF DATA:" +"\n" +pdf_cleaned +
    "\n\nWEBSITE DATA \n" + website_text)
combined_text[:500]
# 2. Chunking the text Using the Jina AI

JINA_SEGMENT_URL = "https://api.jina.ai/v1/segment"

def chunk_with_jina(text: str, max_len: int = 8000):
    """Split text into <=8k chunks and send each to Jina Segmenter API with fallback."""
    headers = {"Authorization": f"Bearer {secret_value_0}"}
    chunks = []

    for i in range(0, len(text), max_len):
        part = text[i:i+max_len]

        payload = {
            "content": part,
            "config": {"split_length": 500}  # ~500 char chunks
        }

        resp = requests.post(JINA_SEGMENT_URL, headers=headers, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            segs = [seg["text"] for seg in data.get("segments", [])]

            if segs:
                chunks.extend(segs)
            else:
                # fallback: just append raw piece if Jina failed
                chunks.append(part)

        else:
            print("Jina Segmenter failed:", resp.status_code, resp.text)
            chunks.append(part)  # fallback

    return chunks


chunk = chunk_with_jina(combined_text)
print(len(chunk))
# 3. Embedding the Chunk Using the Jina AI Embedder


JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"

import numpy as np
import requests

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"

def embed_with_jina(texts, api_key, batch_size=16):
    """
    Convert a list of texts into embeddings using Jina Embeddings API.
    Splits into batches to avoid timeout / payload errors.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = [{"text": t} for t in batch]

        payload = {
            "model": "jina-embeddings-v4",
            "task": "text-matching",   # good for chatbot/retrieval
            "input": inputs
        }

        resp = requests.post(JINA_EMBED_URL, headers=headers, json=payload)

        if resp.status_code == 200:
            data = resp.json()
            vectors = [item["embedding"] for item in data["data"]]
            all_vectors.extend(vectors)
        else:
            raise Exception(f"Embedding failed: {resp.status_code}, {resp.text}")

    return np.array(all_vectors, dtype="float32")


 


embeds = embed_with_jina(chunk, secret_value_0)
print(embeds)
# 4. Indexing By Using FAISS(Facebook AI Similarity Search)

def build_index(vectors):
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

index = build_index(embeds)
index
# 5. Searching fuction Using Jina AI (User Entered Query into Vectors)

def embed_query_jina(query, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "jina-embeddings-v4",
        "task": "text-matching",
        "input": [{"text": query}]
    }
    resp = requests.post(JINA_EMBED_URL, headers=headers, json=payload)

    if resp.status_code == 200:
        data = resp.json()
        vector = data["data"][0]["embedding"]
        return np.array([vector], dtype="float32")   # shape (1, dim)
    else:
        raise Exception(f"Query embedding failed: {resp.status_code}, {resp.text}")
# 6. Searching the Query by User

def search(query, index, chunk, top_k=3):
    q_vec = embed_query_jina(query, secret_value_0)  # Use correct function
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, top_k)
    return [chunk[i] for i in I[0]]
# 7. Reranking the Document for Better and Accurate Results

def rerank_with_jina(query, docs, api_key, top_n=3):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    formatted_docs = [{"text": d} for d in docs]

    data = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": top_n,
        "documents": formatted_docs,
        "return_documents": True
    }

    resp = requests.post("https://api.jina.ai/v1/rerank", headers=headers, json=data)
    out = resp.json()

    if resp.status_code == 200 and "results" in out:
        return [r["document"]["text"] for r in out["results"]]
    else:
        raise Exception(f"Reranking failed: {resp.status_code}, {out}")
# 8. Now, Answering the Question

def answer_question(query, index, chunk, history, api_key, top_k=5, rerank_n=3):
    # Step 1: Retrieve candidates
    retrieved = search(query, index, chunk, top_k=top_k)

    # Step 2:  To do with Reranker
    reranked = rerank_with_jina(query, retrieved, api_key, top_n=rerank_n)

    # Step 3: Build context
    context = "\n".join(reranked)

    # Step 4: Use LLM to answer
    resp = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-70b-instruct",
        messages=[
            {"role": "system", "content": "You are an insurance assistant."},
            *history,
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    answer = resp.choices[0].message.content

    # Update history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})

    return answer
# 9. Defining the Intent of the User

def classify_intent(query):
    q = query.lower().strip()

    
    if q in ["hi", "hello", "hey", "good morning", "good evening"]:
        return "greeting"

    elif any(word in q for word in ["coverage", "benefits", "exclusions", "claim", "policy", "premium", "health insurance"]):
        return "policy_info"

    elif any(word in q for word in ["company", "about", "vision", "history", "branches", "icici"]):
        return "company_info"

    elif any(word in q for word in ["contact", "support", "customer care", "help", "payment"]):
        return "general_support"

    else:
        return "policy_info"


# 10. Remembering the Past Memory (Old Conversation)

def insurance_chatbot(query, chunk, index, company_kb, faq_kb, history):
    intent = classify_intent(query)

    if intent == "greeting":
        return "Hello 👋! I'm your insurance assistant. How can I help you today?"

    elif intent == "policy_info":
        return answer_question(query, index, chunk, history,secret_value_0)

    elif intent == "company_info":
        return company_kb.get("about_company", "Company information not available.")

    elif intent == "general_support":
        return faq_kb.get(query.lower(), "Please contact customer support at 1800-209-9777.")

    else:
        return "I'm here to help with insurance-related queries. Could you rephrase?"


# Converting into Real-time ChatBot

company_kb = {
    "about_company": "ICICI Prudential Life Insurance is one of the leading insurance providers in India, "
                     "offering term plans, health insurance, and savings plans."
}
faq_kb = {
    "how can i contact customer support?": "You can call our toll-free number 1800-209-9777 or email support@iciciprulife.com",
    "how do i pay premium?": "Premiums can be paid online via netbanking, debit/credit card, or through ICICI branches."
}

# Start conversation
history = []


# -----------------------
# 11. Streamlit UI (added at the end, no core code changes)
# -----------------------
import streamlit as st

st.set_page_config(page_title="Insurance Chatbot", layout="wide")
st.title("💬 Insurance Chatbot Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

# Input box for user query
query = st.chat_input("Ask about ICICI insurance...")

if query:
    with st.spinner("Thinking..."):
        response = answer_question(
            query,
            st.session_state.index,
            st.session_state.chunks,
            st.session_state.history,
            secret_value_0
        )

    # Display chat bubbles
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        st.write(response)

# -----------------------
# 12. Preload Documents (first run only)
# -----------------------
if "chunks" not in st.session_state:
    with st.spinner("Loading documents..."):
        with pdfplumber.open("ICICI_Insurance.pdf") as pdf:
            all_text = "".join([page.extract_text() for page in pdf.pages])
        pdf_cleaned = clean_pdf(all_text)

        url = "https://www.icicibank.com/personal-banking/insurance"
        website_text = read_website(url, secret_value_0)

        combined_text = "PDF:\n" + pdf_cleaned + "\n\nWebsite:\n" + website_text

        st.session_state.chunks = chunk_with_jina(combined_text, secret_value_0)
        embeds = embed_with_jina(st.session_state.chunks, secret_value_0)
        st.session_state.index = build_index(embeds)

    st.success("Knowledge base loaded!")

# -----------------------
# 13. Optional Sidebar Features
# -----------------------
with st.sidebar:
    st.header("⚙️ Settings & Info")
    st.write("You are chatting with an **Insurance Assistant** trained on:")
    st.markdown("- 📄 ICICI Insurance PDF\n- 🌐 Official Website")

    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.success("Chat history cleared!")

    st.download_button("⬇️ Download Chat History",
                       "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.history]),
                       file_name="chat_history.txt")

    st.toggle("🌙 Dark Mode (use browser/OS setting)")

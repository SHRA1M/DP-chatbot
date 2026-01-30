import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# --- 1. PAGE CONFIG (Must be first) ---
st.set_page_config(
    page_title="DP Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS FOR IFRAME EMBEDDING ---
st.markdown("""
    <style>
    /* Hide Streamlit branding for cleaner embed */
    #MainMenu, header, footer, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    
    /* Remove default padding for iframe */
    .stApp {
        margin: 0 !important;
    }
    
    .main .block-container {
        padding: 0.5rem 1rem 1rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Compact header */
    .chat-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    
    .chat-header img {
        width: 40px;
        height: 40px;
        border-radius: 8px;
    }
    
    .chat-header-text h4 {
        margin: 0;
        color: #002147;
        font-size: 14px;
    }
    
    .chat-header-text p {
        margin: 0;
        color: #666;
        font-size: 11px;
    }
    
    /* Chat container */
    .stChatFloatingInputContainer {
        bottom: 0 !important;
        background: white !important;
        padding: 10px !important;
        border-top: 1px solid #eee;
    }
    
    /* USER MESSAGE: Right Aligned, Clean White */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
        background-color: #002147 !important;
        color: #ffffff !important;
        border-radius: 16px 16px 4px 16px !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        width: fit-content !important;
        max-width: 85%;
        padding: 10px 14px !important;
        margin-bottom: 8px;
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        color: #ffffff !important;
    }
    
    [data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }
    
    /* ASSISTANT MESSAGE: Left Aligned, Light Grey */
    [data-testid="stChatMessage"]:has(img) {
        background-color: #f1f3f4 !important;
        color: #1a1a1a !important;
        border-radius: 16px 16px 16px 4px !important;
        width: fit-content !important;
        max-width: 85%;
        padding: 10px 14px !important;
        margin-bottom: 8px;
    }
    
    /* Assistant avatar smaller */
    [data-testid="stChatMessage"] img {
        width: 28px !important;
        height: 28px !important;
    }
    
    /* Input styling */
    .stChatInput {
        border-radius: 20px !important;
    }
    
    .stChatInput > div {
        border-radius: 20px !important;
        border: 1px solid #ddd !important;
    }
    
    .stChatInput input {
        font-size: 14px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #002147 !important;
    }
    
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #c1c1c1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #a1a1a1;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD KNOWLEDGE BASE ---
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4})
    except Exception as e:
        st.error(f"Knowledge base error: {e}")
        return None

retriever = load_retriever()

# --- 4. PATHS ---
logo_path = "data/logo_transparent.png"
if not os.path.exists(logo_path):
    logo_path = None

# --- 5. GROQ CLIENT ---
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error(f"⚠️ API Configuration Error")
    st.stop()

# --- 6. SYSTEM INSTRUCTIONS ---
SYSTEM_INSTRUCTIONS = """You are "DP Assistant", the official AI Customer Service Assistant for Digital Protection, a data protection and compliance consultancy in Amman, Jordan.

## PERSONALITY
- Professional, consultative, and friendly
- Never use slang or emojis
- Be concise but helpful

## RESPONSE RULES
1. **Service Questions**: Confirm understanding → Explain with bullets → Mention benefits → Suggest contacting team
2. **Pricing Questions**: Say "Pricing depends on scope and requirements" → Mention models (fixed-price, T&M, retainer) → Direct to contact team
3. **Technical Questions**: Provide info if available → Recommend consultation for details
4. **Contracts/Legal**: Say "I cannot provide contracts or legal advice. Please contact our team directly."

## CRITICAL RULES
- NEVER make up pricing, timelines, or guarantees
- NEVER provide legal advice
- NEVER fix IT issues (printers, wifi, hardware) - we only do Cybersecurity & Compliance
- If user writes in Arabic, respond in English and mention Arabic support available via direct contact
- Always offer to connect with the team: info@dp-technologies.net or +962 790 552 879

## CONTACT INFO
- Email: info@dp-technologies.net
- Phone: +962 790 552 879
- Location: Amman, Jordan"""

# --- 7. INITIALIZE CHAT ---
def get_greeting():
    return """Hello! 👋 Welcome to Digital Protection.

I'm your DP Assistant, here to help you with your questions.
How can I assist you today?"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": get_greeting()}
    ]

# --- 8. COMPACT HEADER (for embedded view) ---
# Check if embedded via URL parameter
query_params = st.query_params
is_embedded = query_params.get("embed", "false").lower() == "true"

if not is_embedded:
    # Show header only if NOT embedded (direct access)
    col1, col2 = st.columns([1, 5])
    with col1:
        if logo_path:
            st.image(logo_path, width=50)
    with col2:
        st.markdown("### Digital Protection Support")

# --- 9. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar = logo_path if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --- 10. HANDLE USER INPUT ---
if prompt := st.chat_input("Type your message..."):
    
    # Check retriever
    if retriever is None:
        st.error("⚠️ Knowledge base unavailable. Please try again later.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar=logo_path):
        with st.spinner("Thinking..."):
            
            # Search knowledge base
            try:
                search_results = retriever.invoke(prompt)
                context = "\n".join([doc.page_content for doc in search_results])
            except:
                context = ""
            
            # Build prompt
            full_prompt = f"""{SYSTEM_INSTRUCTIONS}

---
KNOWLEDGE BASE CONTEXT:
{context}
---

USER QUESTION: {prompt}

ASSISTANT RESPONSE (be concise, professional, no emojis):"""

            try:
                # Call Groq
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": full_prompt}],
                    model="llama3-8b-8192",
                    temperature=0.7,
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean up response (remove any "Dear Customer" etc.)
                unwanted_starts = ["Dear", "Subject:", "Hello,", "Hi,"]
                for start in unwanted_starts:
                    if answer.startswith(start):
                        answer = answer.split("\n", 1)[-1].strip()
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = "I apologize, but I'm having trouble connecting right now. Please try again or contact us directly at info@dp-technologies.net"
                st.markdown(error_msg)

                st.session_state.messages.append({"role": "assistant", "content": error_msg})


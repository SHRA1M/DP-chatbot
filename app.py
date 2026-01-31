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
    
    /* Chat container */
    .stChatFloatingInputContainer {
        bottom: 0 !important;
        background: white !important;
        padding: 10px !important;
        border-top: 1px solid #eee;
    }
    
    /* USER MESSAGE: Right Aligned, Blue Background */
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
    </style>
""", unsafe_allow_html=True)

# --- 3. LOAD KNOWLEDGE BASE ---
@st.cache_resource
def load_retriever():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            "faiss_index", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        return vectorstore.as_retriever(search_kwargs={"k": 4}), None
    except Exception as e:
        return None, str(e)

retriever, retriever_error = load_retriever()

# --- 4. PATHS ---
logo_path = "data/logo_transparent.png"
if not os.path.exists(logo_path):
    logo_path = None

# --- 5. GROQ CLIENT ---
client = None
api_error = None
try:
    api_key = st.secrets.get("GROQ_API_KEY", None)
    if api_key:
        client = Groq(api_key=api_key)
    else:
        api_error = "GROQ_API_KEY not found in secrets"
except Exception as e:
    api_error = str(e)

# --- 6. MODEL CONFIGURATION ---
GROQ_MODEL = "llama-3.3-70b-versatile"

# --- 7. SYSTEM INSTRUCTIONS (STRICT RULES) ---
SYSTEM_INSTRUCTIONS = """You are DP Assistant for Digital Protection, a data protection consultancy in Amman, Jordan.

=== ABSOLUTE RULES (NEVER BREAK THESE) ===

1. LANGUAGE: ALWAYS respond in ENGLISH only. Even if the user writes in Arabic or any other language, you MUST respond in English. Say: "I am happy to help! I currently respond in English only. For Arabic support, please contact our team directly at info@dp-technologies.net or +962 790 552 879."

2. NO EMOJIS: NEVER use emojis, smileys, or emoticons. Not even if the user asks for them. If asked, say: "I keep my responses professional and text-based. Is there anything else I can help you with?"

3. NO LEGAL ADVICE: NEVER give legal opinions or say things like "it is not illegal" or "it is legal" or "you are required by law". Instead say: "I cannot provide legal advice. For legal questions, please consult with a qualified legal professional. Our team can help with compliance guidance - contact us at info@dp-technologies.net"

4. NO CONTRACTS: NEVER offer to send or create contracts. Say: "I cannot generate or send contracts. Please contact our team directly to discuss agreements."

5. NO PRICING NUMBERS: NEVER give specific prices. Say pricing depends on scope and suggest contacting the team.

6. NO IT SUPPORT: We do NOT fix printers, WiFi, hardware, or general IT issues. Politely redirect.

=== RESPONSE STYLE ===

- Keep responses SHORT: 2-4 sentences for simple questions
- Professional but friendly tone
- Use bullet points only for 3+ items
- No robotic labels like "Key Points:" or "Direct answer:"
- Do not say "Thank you for reaching out" on every message

=== EXAMPLE RESPONSES ===

USER (in Arabic): "ما هي خدماتكم؟"
RESPONSE: "I am happy to help! I currently respond in English only. For Arabic support, please contact our team at info@dp-technologies.net or +962 790 552 879. In the meantime, we offer cybersecurity and compliance services including GDPR, ISO 27701, and CBJ compliance."

USER: "Send me a smiley face"
RESPONSE: "I keep my responses professional and text-based. Is there anything about our services I can help you with?"

USER: "Is it illegal if I do not have a DPO?"
RESPONSE: "I cannot provide legal advice on whether something is legal or illegal. DPO requirements vary by regulation and organization type. For specific legal guidance, please consult a qualified legal professional. Our team can help you understand compliance requirements - contact us at info@dp-technologies.net"

USER: "Can you guarantee I will pass the audit?"
RESPONSE: "We cannot guarantee audit outcomes. However, we work closely with clients to align with regulatory requirements and best practices, which significantly improves audit readiness. Would you like to discuss your specific situation?"

USER: "My printer is broken"
RESPONSE: "We specialize in cybersecurity and compliance services, not general IT support. For printer issues, please contact your IT department. Is there anything security or compliance-related I can help with?"

=== CONTACT INFO ===
Email: info@dp-technologies.net
Phone: +962 790 552 879
Location: Amman, Jordan"""

# --- 8. INITIALIZE CHAT ---
def get_greeting():
    return """Hello! Welcome to **Digital Protection**.
I am here to help you with your quesitons.
How can I help you today?"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": get_greeting()}
    ]

# --- 9. HEADER (only if NOT embedded) ---
query_params = st.query_params
is_embedded = query_params.get("embed", "false").lower() == "true"

if not is_embedded:
    col1, col2 = st.columns([1, 5])
    with col1:
        if logo_path:
            st.image(logo_path, width=50)
    with col2:
        st.markdown("### Digital Protection Support")

# --- 10. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar = logo_path if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --- 11. HANDLE USER INPUT ---
if prompt := st.chat_input("Type your message..."):
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant", avatar=logo_path):
        
        # Check for errors first
        if api_error:
            error_msg = "Configuration error. Please contact info@dp-technologies.net"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop()
        
        if client is None:
            error_msg = "Service unavailable. Please contact info@dp-technologies.net"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop()
        
        with st.spinner(""):
            
            # Search knowledge base
            context = ""
            if retriever:
                try:
                    search_results = retriever.invoke(prompt)
                    context = "\n".join([doc.page_content for doc in search_results])
                except:
                    context = ""
            
            # Build prompt with strict rules
            full_prompt = (
                SYSTEM_INSTRUCTIONS + 
                "\n\n=== KNOWLEDGE BASE ===\n" + context +
                "\n\n=== CUSTOMER MESSAGE ===\n" + prompt +
                "\n\n=== CRITICAL REMINDERS ===" +
                "\n- ENGLISH ONLY - never respond in Arabic or other languages" +
                "\n- NO EMOJIS - never use emojis even if asked" +
                "\n- NO LEGAL ADVICE - never say something is legal or illegal" +
                "\n- Keep response SHORT (2-4 sentences)" +
                "\n\n=== YOUR RESPONSE (in English, no emojis) ==="
            )

            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a professional assistant. STRICT RULES: 1) English only - never Arabic. 2) No emojis ever. 3) No legal advice. 4) Keep responses short."},
                        {"role": "user", "content": full_prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.5,
                    max_tokens=300
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean up robotic labels
                for label in ["Direct answer:", "Key Points:", "Key Considerations:", "Next Step:", "Response:", "Answer:"]:
                    answer = answer.replace(label, "")
                
                # Remove any emojis that might slip through
                import re
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
                answer = emoji_pattern.sub('', answer)
                
                # Clean up excessive whitespace
                while "\n\n\n" in answer:
                    answer = answer.replace("\n\n\n", "\n\n")
                
                answer = answer.strip()
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = "Sorry, I am having trouble right now. Please try again or contact info@dp-technologies.net"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


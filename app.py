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

# --- 7. SYSTEM INSTRUCTIONS (CONCISE & NATURAL) ---
SYSTEM_INSTRUCTIONS = """You are DP Assistant for Digital Protection, a data protection consultancy in Amman, Jordan.

RESPONSE LENGTH RULES (VERY IMPORTANT):
- Keep responses SHORT - 2 to 4 sentences for simple questions
- Maximum 5-6 bullet points for lists
- Only give detailed responses when the question requires it
- Do not over-explain. Be concise.

TONE:
- Professional but friendly
- Talk like a helpful human, not a robot
- Do not say Thank you for reaching out on every message - only on first interaction or when appropriate
- Do not repeat contact info unless asked or truly relevant

FORMATTING:
- Use bullet points only when listing 3+ items
- No labels like Direct answer or Key Points
- Bold only key terms, not everything

EXAMPLE RESPONSES:

Q: What services do you offer?
A: We specialize in cybersecurity and compliance services:
- **Privacy & Compliance** - GDPR, ISO 27701, CBJ
- **Security Assessments** - Vulnerability scanning, risk analysis
- **Network Security** - Firewalls, WAF
- **Identity & Access Management** - IAM/PAM solutions

Would you like details on any specific service?

Q: Do you work with banks?
A: Yes, we work extensively with banks and financial institutions. We have expertise in CBJ compliance and SAMA requirements. Let me know if you would like to discuss your specific needs.

Q: How much does it cost?
A: Pricing depends on the scope and complexity of your project. We offer fixed-price, time and materials, and retainer options. Contact us at info@dp-technologies.net for a customized quote.

Q: Can you fix my printer?
A: We specialize in cybersecurity and compliance, not general IT support. For printer issues, please contact your IT department. Is there anything security or compliance-related I can help with?

Q: What is GDPR?
A: GDPR is the General Data Protection Regulation - an EU law that governs how organizations handle personal data of EU residents. It applies globally to any company processing EU citizens data. Need help with GDPR compliance?

RULES:
- NEVER make up pricing or timelines
- NEVER give legal advice
- For IT support requests, politely redirect
- Contact: info@dp-technologies.net | +962 790 552 879"""

# --- 8. INITIALIZE CHAT ---
def get_greeting():
    return """Hello! Welcome to **Digital Protection**.
I am here to help you with your questions.
What can I help you with?"""

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
            
            # Count messages to know if it is early in conversation
            msg_count = len(st.session_state.messages)
            
            # Build prompt
            full_prompt = (
                SYSTEM_INSTRUCTIONS + 
                "\n\nKNOWLEDGE BASE:\n" + context +
                "\n\nCONVERSATION CONTEXT: This is message #" + str(msg_count) + " in the conversation." +
                "\n\nCUSTOMER: " + prompt +
                "\n\nIMPORTANT:" +
                "\n- Keep your response SHORT and concise" +
                "\n- 2-4 sentences for simple questions" +
                "\n- Do not start every response with Thank you" +
                "\n- Do not repeat contact info unless needed" +
                "\n- Answer directly, then offer to help further if appropriate" +
                "\n\nRESPONSE:"
            )

            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a concise, helpful assistant. Keep responses short - 2-4 sentences for simple questions. No unnecessary fluff."},
                        {"role": "user", "content": full_prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.6,
                    max_tokens=300
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean up robotic labels
                for label in ["Direct answer:", "Key Points:", "Key Considerations:", "Next Step:", "Response:", "Answer:"]:
                    answer = answer.replace(label, "")
                
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


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

# --- 7. SYSTEM INSTRUCTIONS (IMPROVED FOR FORMATTING) ---
SYSTEM_INSTRUCTIONS = """You are "DP Assistant", the official AI Customer Service Assistant for Digital Protection, a specialized data protection and compliance consultancy based in Amman, Jordan.

## YOUR IDENTITY
- Name: DP Assistant
- Company: Digital Protection
- Location: Amman, Jordan
- Contact: info@dp-technologies.net | +962 790 552 879

## PERSONALITY & TONE
- Professional and consultative
- Friendly but business-appropriate
- NO slang, NO emojis, NO casual language
- Speak like an expert consultant

## MANDATORY FORMATTING RULES (FOLLOW STRICTLY)

### For Service/Product Questions:
Structure your response EXACTLY like this:
1. One-line acknowledgment
2. Brief explanation (1-2 sentences)
3. **Key Points:** (use bullet points)
   - Point 1
   - Point 2
   - Point 3
4. **Next Step:** Contact our team for detailed discussion

### For General Questions:
Structure your response EXACTLY like this:
1. Direct answer (1-2 sentences)
2. **Details:** (if needed, use bullets)
   - Detail 1
   - Detail 2
3. Offer further assistance

### For Pricing Questions:
"Pricing depends on the scope and specific requirements of your project.

**Our Pricing Models:**
- **Fixed-price:** For projects with defined deliverables
- **Time & Materials:** For advisory services
- **Retainer:** For ongoing support

**Next Step:** Contact us at info@dp-technologies.net or +962 790 552 879 for a customized quote."

### For IT Support Requests (printers, wifi, hardware):
"I appreciate you reaching out. However, Digital Protection specializes exclusively in:
- **Cybersecurity Services**
- **Compliance & Governance** (GDPR, ISO, CBJ)
- **Data Protection Consulting**

We do not provide general IT support. For hardware or network issues, please contact your IT department or a local IT service provider.

Is there anything related to cybersecurity or compliance I can help you with?"

## RESPONSE RULES
1. ALWAYS use **bold** for headers and key terms
2. ALWAYS use bullet points (-) for lists
3. Keep responses concise (under 150 words when possible)
4. NEVER write long paragraphs - break them up
5. NEVER start with "I'm happy to help" or similar filler phrases
6. NEVER make up pricing, timelines, or guarantees
7. NEVER provide legal advice - refer to the team
8. If user writes in Arabic, respond in English and mention Arabic support via direct contact

## EXAMPLE GOOD RESPONSE:

User: "What services do you offer?"

Response:
"Digital Protection provides specialized cybersecurity and compliance services.

**Our Core Services:**
- **Privacy & Compliance:** GDPR, ISO 27701, CBJ compliance assessments
- **Security Assessments:** Vulnerability analysis and risk evaluation
- **Network Security:** Firewalls, WAF, secure infrastructure
- **Identity & Access:** IAM and PAM solutions

**Industries We Serve:**
- Banking & Financial Services
- Healthcare
- Government
- Telecommunications

**Next Step:** Contact us at info@dp-technologies.net to discuss your specific needs."

## CONTACT INFORMATION (Always available to share)
- **Email:** info@dp-technologies.net
- **Phone:** +962 790 552 879
- **Location:** Amman, Jordan
- **Response Time:** Within one business day"""

# --- 8. INITIALIZE CHAT ---
def get_greeting():
    return """Hello! Welcome to **Digital Protection**.

I'm your DP Assistant, here to help you with your questions. 
How can I assist you today?"""

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
            error_msg = f"Configuration error: {api_error}. Please contact support."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop()
        
        if client is None:
            error_msg = "API not configured. Please contact support."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.stop()
        
        with st.spinner("Thinking..."):
            
            # Search knowledge base (if available)
            context = ""
            if retriever:
                try:
                    search_results = retriever.invoke(prompt)
                    context = "\n".join([doc.page_content for doc in search_results])
                except Exception as e:
                    context = ""
            
            # Build prompt with strong formatting instructions
            full_prompt = f"""SYSTEM INSTRUCTIONS:
{SYSTEM_INSTRUCTIONS}

---
KNOWLEDGE BASE (Use this information to answer accurately):
{context}
---

USER QUESTION: {prompt}

IMPORTANT REMINDERS BEFORE YOU RESPOND:
1. Use **bold** for headers and key terms
2. Use bullet points (-) for any list of items
3. Keep it concise and professional
4. Structure your response clearly
5. Do NOT write long paragraphs

YOUR RESPONSE:"""

            try:
                # Call Groq
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a professional customer service assistant. Always format responses with bullet points and bold headers. Never write long paragraphs."},
                        {"role": "user", "content": full_prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.5,  # Lower temperature for more consistent formatting
                    max_tokens=500
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean up response - remove unwanted openings
                unwanted_starts = [
                    "Dear", "Subject:", "Hello,", "Hi,", "I'm happy to help",
                    "Thank you for reaching out", "Thanks for your question",
                    "Great question", "Good question"
                ]
                for start in unwanted_starts:
                    if answer.lower().startswith(start.lower()):
                        lines = answer.split("\n", 1)
                        if len(lines) > 1:
                            answer = lines[1].strip()
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"I encountered an error. Please try again or contact us at info@dp-technologies.net"})

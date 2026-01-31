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

# --- 7. SYSTEM INSTRUCTIONS (NATURAL CONVERSATIONAL TONE) ---
SYSTEM_INSTRUCTIONS = """You are "DP Assistant", the friendly and professional AI Customer Service Assistant for Digital Protection, a specialized data protection and compliance consultancy based in Amman, Jordan.

## YOUR ROLE
You are here to HELP customers. You genuinely care about their needs. You are knowledgeable, warm, and professional - like a helpful consultant who wants to guide them to the right solution.

## YOUR PERSONALITY
- Warm and welcoming, but professional
- Helpful and solution-oriented
- Patient and understanding
- Knowledgeable but not arrogant
- You use "please", "thank you", "I'd be happy to help"
- You speak like a real person, not a robot

## TONE EXAMPLES
- "I'd be happy to help you with that!"
- "Great question! Let me explain..."
- "Absolutely, we can assist with that."
- "I understand your concern. Here's what I recommend..."
- "Please feel free to reach out to our team..."
- "Thank you for your interest in our services!"

## FORMATTING RULES (CRITICAL)

### NEVER USE THESE ROBOTIC LABELS:
- ❌ "Direct answer:"
- ❌ "Key Points:"
- ❌ "Key Considerations:"
- ❌ "Next Step:" (as a label at the start of a line)
- ❌ Numbered lists like "1. First thing 2. Second thing"

### INSTEAD, WRITE NATURALLY:
- ✅ Use bullet points for lists (but without labels)
- ✅ Write in flowing, natural sentences
- ✅ Bold **important terms** only, not headers
- ✅ End with a friendly invitation to contact us

### GOOD RESPONSE EXAMPLE:

User: "What services do you offer?"

Response:
"Thank you for your interest in Digital Protection!

We specialize in helping organizations achieve and maintain data protection compliance. Our main services include:

- **Privacy & Compliance** - GDPR, ISO 27701, and CBJ compliance assessments, policy development, and audit preparation
- **Security Assessments** - Vulnerability scanning, risk analysis, and remediation guidance
- **Network & Application Security** - Firewalls, WAF, and secure infrastructure solutions
- **Identity & Access Management** - IAM and PAM solutions to enforce security controls

We work with organizations across banking, healthcare, government, telecom, and other industries throughout the MENA region.

I'd be happy to arrange a consultation to discuss your specific needs. Please contact us at info@dp-technologies.net or call +962 790 552 879."

### ANOTHER GOOD EXAMPLE:

User: "How much does GDPR compliance cost?"

Response:
"Great question! The cost of GDPR compliance varies depending on your organization's size, current compliance status, and specific requirements.

We offer flexible pricing models to suit different needs:

- **Fixed-price projects** for defined deliverables like gap assessments
- **Time & materials** for ongoing advisory work
- **Retainer arrangements** for continuous support

To give you an accurate quote, we'd need to understand your situation better. I'd recommend scheduling a free initial consultation with our team.

Please reach out to us at info@dp-technologies.net or +962 790 552 879, and we'll be happy to discuss your requirements."

### FOR IT SUPPORT REQUESTS (printers, wifi, hardware):

"Thank you for reaching out! I appreciate you contacting us.

However, Digital Protection specializes specifically in **cybersecurity and compliance services** - things like GDPR compliance, security assessments, and data protection consulting.

For printer, WiFi, or general IT support, I'd recommend contacting your internal IT department or a local IT service provider.

If you have any questions about cybersecurity or compliance, I'm here to help! Is there anything in that area I can assist you with?"

## CRITICAL RULES
1. ALWAYS be warm and helpful - you WANT to assist customers
2. ALWAYS use "please" and offer to help further
3. ALWAYS end with contact information when relevant
4. NEVER use robotic labels like "Direct answer:" or "Key Points:"
5. NEVER make up pricing, timelines, or guarantees
6. NEVER provide legal advice - politely refer to the team
7. If asked about IT support (printers, wifi), politely explain we only do cybersecurity/compliance
8. If user writes in Arabic, respond in English and mention Arabic support is available by contacting the team directly

## CONTACT INFORMATION
- Email: info@dp-technologies.net
- Phone: +962 790 552 879
- Location: Amman, Jordan
- Hours: Sunday-Thursday, 9 AM - 6 PM (Jordan Time)
- Response Time: Within one business day"""

# --- 8. INITIALIZE CHAT ---
def get_greeting():
    return """Hello! Welcome to **Digital Protection** 👋
I'm your DP Assistant, and I'm here to help you with any questions. 
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
        
        with st.spinner(""):
            
            # Search knowledge base (if available)
            context = ""
            if retriever:
                try:
                    search_results = retriever.invoke(prompt)
                    context = "\n".join([doc.page_content for doc in search_results])
                except Exception as e:
                    context = ""
            
            # Build prompt
            full_prompt = f"""{SYSTEM_INSTRUCTIONS}

---
COMPANY KNOWLEDGE BASE (Use this to answer accurately):
{context}
---

CUSTOMER MESSAGE: {prompt}

REMEMBER:
- Be warm, helpful, and professional
- Write naturally like a real customer service rep
- Use bullet points for lists (NO labels like "Key Points:")
- NEVER start with "Direct answer:" or similar labels
- Include contact info when helpful
- Say "please" and offer to help further

YOUR RESPONSE:"""

            try:
                # Call Groq
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a warm, helpful customer service assistant. Write naturally and conversationally. Never use robotic labels like 'Direct answer:' or 'Key Points:'. Be professional but friendly."},
                        {"role": "user", "content": full_prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.6,
                    max_tokens=600
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean up any robotic labels that might slip through
                robotic_labels = [
                    "Direct answer:", "Direct Answer:",
                    "Key Points:", "Key points:",
                    "Key Considerations:", "Key considerations:",
                    "Next Step:", "Next step:",
                    "Response:", "Answer:",
                    "Here is my response:", "Here's my response:"
                ]
                for label in robotic_labels:
                    answer = answer.replace(label, "")
                
                # Clean up any double line breaks
                while "\n\n\n" in answer:
                    answer = answer.replace("\n\n\n", "\n\n")
                
                answer = answer.strip()
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = "I apologize, but I'm having a little trouble right now. Please try again, or feel free to contact our team directly at info@dp-technologies.net or +962 790 552 879."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

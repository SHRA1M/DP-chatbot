import streamlit as st
import os
import re
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

# --- 7. SYSTEM INSTRUCTIONS (BILINGUAL) ---
SYSTEM_INSTRUCTIONS_EN = """You are DP Assistant for Digital Protection, a data protection consultancy in Amman, Jordan.

=== LANGUAGE RULE ===
The user is writing in ENGLISH. You MUST respond in ENGLISH only.

=== ABSOLUTE RULES ===
1. NO EMOJIS: Never use emojis even if asked.
2. NO LEGAL ADVICE: Never say something is "legal" or "illegal". Say: "I cannot provide legal advice. Please consult a qualified legal professional."
3. NO CONTRACTS: Never offer to send contracts. Say: "I cannot generate contracts. Please contact our team directly."
4. NO PRICING NUMBERS: Never give specific prices. Say pricing depends on scope.
5. NO IT SUPPORT: We do NOT fix printers, WiFi, or hardware.

=== RESPONSE STYLE ===
- Keep responses SHORT: 2-4 sentences for simple questions
- Professional but friendly
- Use bullet points only for 3+ items

=== CONTACT INFO ===
Email: info@dp-technologies.net
Phone: +962 790 552 879
Location: Amman, Jordan"""

SYSTEM_INSTRUCTIONS_AR = """انت مساعد DP لشركة Digital Protection، وهي شركة استشارات لحماية البيانات في عمان، الاردن.

=== قاعدة اللغة ===
المستخدم يكتب بالعربية. يجب ان ترد بالعربية فقط.

=== القواعد المطلقة ===
1. بدون رموز تعبيرية: لا تستخدم الايموجي ابدا حتى لو طلب منك.
2. بدون استشارات قانونية: لا تقل ابدا ان شيئا "قانوني" او "غير قانوني". قل: "لا استطيع تقديم استشارات قانونية. يرجى استشارة محام مختص."
3. بدون عقود: لا تعرض ابدا ارسال عقود. قل: "لا استطيع انشاء عقود. يرجى التواصل مع فريقنا مباشرة."
4. بدون ارقام اسعار: لا تعطي اسعارا محددة ابدا. قل ان التسعير يعتمد على نطاق المشروع.
5. بدون دعم تقني عام: نحن لا نصلح الطابعات او الواي فاي او الاجهزة.

=== اسلوب الرد ===
- اجعل الردود قصيرة: 2-4 جمل للاسئلة البسيطة
- مهني ولكن ودود
- استخدم النقاط فقط عند وجود 3 عناصر او اكثر

=== خدماتنا ===
- الخصوصية والامتثال: GDPR، ISO 27701، متطلبات البنك المركزي الاردني
- تقييمات الامن: فحص الثغرات وتحليل المخاطر
- امن الشبكات: جدران الحماية، WAF
- ادارة الهوية والوصول: حلول IAM و PAM

=== معلومات الاتصال ===
البريد الالكتروني: info@dp-technologies.net
الهاتف: +962 790 552 879
الموقع: عمان، الاردن"""

# --- 8. HELPER FUNCTION: Detect Arabic ---
def is_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    arabic_chars = len(arabic_pattern.findall(text))
    return arabic_chars > 0

# --- 9. INITIALIZE CHAT ---
def get_greeting():
    return """Hello! Welcome to **Digital Protection**.

مرحبا! اهلا بك في **Digital Protection**.

I am here to help with questions about compliance, security, and data protection.

انا هنا لمساعدتك في اسئلة الامتثال والامن وحماية البيانات.

How can I help you? | كيف يمكنني مساعدتك؟"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": get_greeting()}
    ]

# --- 10. HEADER (only if NOT embedded) ---
query_params = st.query_params
is_embedded = query_params.get("embed", "false").lower() == "true"

if not is_embedded:
    col1, col2 = st.columns([1, 5])
    with col1:
        if logo_path:
            st.image(logo_path, width=50)
    with col2:
        st.markdown("### Digital Protection Support")

# --- 11. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar = logo_path if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# --- 12. HANDLE USER INPUT ---
if prompt := st.chat_input("Type your message... | اكتب رسالتك..."):
    
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
            
            # Detect language
            user_speaks_arabic = is_arabic(prompt)
            
            # Select appropriate instructions
            if user_speaks_arabic:
                system_instructions = SYSTEM_INSTRUCTIONS_AR
                language_reminder = "رد بالعربية فقط. اجعل الرد قصيرا (2-4 جمل). بدون رموز تعبيرية."
            else:
                system_instructions = SYSTEM_INSTRUCTIONS_EN
                language_reminder = "Respond in English only. Keep response short (2-4 sentences). No emojis."
            
            # Build prompt
            full_prompt = (
                system_instructions + 
                "\n\n=== KNOWLEDGE BASE ===\n" + context +
                "\n\n=== CUSTOMER MESSAGE ===\n" + prompt +
                "\n\n=== REMINDER ===\n" + language_reminder +
                "\n\n=== YOUR RESPONSE ==="
            )

            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": f"You are a bilingual assistant (English/Arabic). {language_reminder}"},
                        {"role": "user", "content": full_prompt}
                    ],
                    model=GROQ_MODEL,
                    temperature=0.5,
                    max_tokens=350
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Clean up robotic labels
                for label in ["Direct answer:", "Key Points:", "Key Considerations:", "Next Step:", "Response:", "Answer:", "الاجابة:", "النقاط الرئيسية:"]:
                    answer = answer.replace(label, "")
                
                # Remove any emojis
                emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
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
                if user_speaks_arabic:
                    error_msg = "عذرا، حدث خطا. يرجى المحاولة مرة اخرى او التواصل معنا على info@dp-technologies.net"
                else:
                    error_msg = "Sorry, an error occurred. Please try again or contact info@dp-technologies.net"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

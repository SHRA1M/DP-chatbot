import streamlit as st
import os
import re
import time
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

# --- 2. CUSTOM CSS ---
st.markdown("""
    <style>
    #MainMenu, header, footer, .stDeployButton {
        visibility: hidden !important;
        display: none !important;
    }
    
    .stApp { margin: 0 !important; }
    
    .main .block-container {
        padding: 0.5rem 1rem 1rem 1rem !important;
        max-width: 100% !important;
    }
    
    .stChatFloatingInputContainer {
        bottom: 0 !important;
        background: white !important;
        padding: 10px !important;
        border-top: 1px solid #eee;
    }
    
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
    
    [data-testid="stChatMessageAvatarUser"] { display: none !important; }
    
    [data-testid="stChatMessage"]:has(img) {
        background-color: #f1f3f4 !important;
        color: #1a1a1a !important;
        border-radius: 16px 16px 16px 4px !important;
        width: fit-content !important;
        max-width: 85%;
        padding: 10px 14px !important;
        margin-bottom: 8px;
    }
    
    [data-testid="stChatMessage"] img {
        width: 28px !important;
        height: 28px !important;
    }
    
    .stChatInput { border-radius: 20px !important; }
    .stChatInput > div { border-radius: 20px !important; border: 1px solid #ddd !important; }
    [data-testid="stSidebar"] { display: none !important; }
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #f1f1f1; }
    ::-webkit-scrollbar-thumb { background: #c1c1c1; border-radius: 3px; }
    
    .arabic-text { direction: rtl; text-align: right; }
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
BACKUP_MODEL = "llama-3.1-8b-instant"  # Faster backup model

# --- 7. GREETINGS ---
GREETING_EN = """Hello! Welcome to **Digital Protection**.

I am here to help you with your questions.

How can I help you?"""

GREETING_AR = """<div class="arabic-text">

مرحبا! اهلا بك في **Digital Protection**.

انا هنا لمساعدتك في اسئلتك.

كيف يمكنني مساعدتك؟

</div>"""

# --- 8. SYSTEM INSTRUCTIONS ---
SYSTEM_INSTRUCTIONS_EN = """You are DP Assistant for Digital Protection, a data protection consultancy in Amman, Jordan.

LANGUAGE: Respond in ENGLISH only.

RULES:
1. NO EMOJIS ever
2. NO LEGAL ADVICE - say "I cannot provide legal advice. Please consult a qualified legal professional."
3. NO CONTRACTS - say "I cannot generate contracts. Please contact our team."
4. NO SPECIFIC PRICES - say pricing depends on scope
5. NO IT SUPPORT for printers, WiFi, hardware

STYLE: Keep responses SHORT (2-4 sentences). Professional but friendly.

SERVICES:
- Privacy & Compliance: GDPR, ISO 27701, CBJ
- Security Assessments: Vulnerability scanning, risk analysis
- Network Security: Firewalls, WAF
- Identity & Access Management: IAM/PAM

CONTACT: info@dp-technologies.net | +962 790 552 879 | Amman, Jordan"""

SYSTEM_INSTRUCTIONS_AR = """انت مساعد DP لشركة Digital Protection في عمان، الاردن.

اللغة: رد بالعربية فقط.

القواعد:
1. بدون رموز تعبيرية ابدا
2. بدون استشارات قانونية - قل "لا استطيع تقديم استشارات قانونية. يرجى استشارة محام مختص."
3. بدون عقود - قل "لا استطيع انشاء عقود. يرجى التواصل مع فريقنا."
4. بدون اسعار محددة - قل التسعير يعتمد على نطاق المشروع
5. بدون دعم تقني للطابعات والواي فاي

الاسلوب: ردود قصيرة (2-4 جمل). مهني وودود.

الخدمات:
- الخصوصية والامتثال: GDPR، ISO 27701، البنك المركزي الاردني
- تقييمات الامن: فحص الثغرات، تحليل المخاطر
- امن الشبكات: جدران الحماية، WAF
- ادارة الهوية والوصول: IAM/PAM

التواصل: info@dp-technologies.net | +962 790 552 879 | عمان، الاردن"""

# --- 9. HELPER FUNCTIONS ---
def is_arabic(text):
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return bool(arabic_pattern.search(text))

def clean_response(answer, is_arabic_response=False):
    """Clean up the response text"""
    # Remove robotic labels
    labels = ["Direct answer:", "Key Points:", "Key Considerations:", "Next Step:", 
              "Response:", "Answer:", "الاجابة:", "النقاط الرئيسية:", "الخطوة التالية:"]
    for label in labels:
        answer = answer.replace(label, "")
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    answer = emoji_pattern.sub('', answer)
    
    # Clean whitespace
    while "\n\n\n" in answer:
        answer = answer.replace("\n\n\n", "\n\n")
    
    answer = answer.strip()
    
    # Wrap Arabic in RTL div
    if is_arabic_response:
        answer = f'<div class="arabic-text">{answer}</div>'
    
    return answer

def call_groq_with_retry(client, messages, model, max_retries=3):
    """Call Groq API with retry logic"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=0.5,
                max_tokens=350,
                timeout=30  # 30 second timeout
            )
            return response, None
        except Exception as e:
            last_error = str(e)
            
            # If rate limited, wait before retry
            if "rate" in last_error.lower() or "limit" in last_error.lower():
                time.sleep(2 * (attempt + 1))  # Exponential backoff: 2s, 4s, 6s
            elif "timeout" in last_error.lower():
                time.sleep(1)
            else:
                # For other errors, try backup model on second attempt
                if attempt == 1 and model != BACKUP_MODEL:
                    model = BACKUP_MODEL
                time.sleep(1)
    
    return None, last_error

# --- 10. FALLBACK RESPONSES ---
FALLBACK_EN = {
    "services": "We offer cybersecurity and compliance services including GDPR, ISO 27701, CBJ compliance, security assessments, and identity management. Contact us at info@dp-technologies.net for details.",
    "pricing": "Pricing depends on the scope of your project. We offer fixed-price, time and materials, and retainer options. Contact info@dp-technologies.net for a quote.",
    "location": "We are located in Amman, Jordan. Contact us at info@dp-technologies.net or +962 790 552 879.",
    "default": "Thank you for your message. For detailed assistance, please contact our team at info@dp-technologies.net or +962 790 552 879."
}

FALLBACK_AR = {
    "services": "نقدم خدمات الامن السيبراني والامتثال بما في ذلك GDPR و ISO 27701 والبنك المركزي الاردني وتقييمات الامن. تواصل معنا على info@dp-technologies.net",
    "pricing": "التسعير يعتمد على نطاق مشروعك. نقدم خيارات السعر الثابت والوقت والمواد والاشتراك. تواصل معنا للحصول على عرض سعر.",
    "location": "نحن في عمان، الاردن. تواصل معنا على info@dp-technologies.net او +962 790 552 879",
    "default": "شكرا لرسالتك. للمساعدة التفصيلية، يرجى التواصل مع فريقنا على info@dp-technologies.net او +962 790 552 879"
}

def get_fallback_response(prompt, is_arabic_lang):
    """Get a fallback response when API fails"""
    prompt_lower = prompt.lower()
    fallback = FALLBACK_AR if is_arabic_lang else FALLBACK_EN
    
    if any(word in prompt_lower for word in ["service", "خدم", "offer", "تقدم"]):
        return fallback["services"]
    elif any(word in prompt_lower for word in ["price", "cost", "سعر", "تكلف", "كم"]):
        return fallback["pricing"]
    elif any(word in prompt_lower for word in ["where", "location", "اين", "موقع"]):
        return fallback["location"]
    else:
        return fallback["default"]

# --- 11. INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"

if "greeting_shown" not in st.session_state:
    st.session_state.greeting_shown = False

if "error_count" not in st.session_state:
    st.session_state.error_count = 0

# --- 12. HEADER WITH LANGUAGE TOGGLE ---
query_params = st.query_params
is_embedded = query_params.get("embed", "false").lower() == "true"

if not is_embedded:
    col1, col2, col3 = st.columns([1, 4, 2])
    with col1:
        if logo_path:
            st.image(logo_path, width=50)
    with col2:
        st.markdown("### Digital Protection Support")
    with col3:
        if st.session_state.ui_language == "en":
            if st.button("بالعربية", key="lang_toggle"):
                st.session_state.ui_language = "ar"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()
        else:
            if st.button("English", key="lang_toggle"):
                st.session_state.ui_language = "en"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()
else:
    col1, col2 = st.columns([5, 1])
    with col2:
        if st.session_state.ui_language == "en":
            if st.button("عربي", key="lang_toggle_embed"):
                st.session_state.ui_language = "ar"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()
        else:
            if st.button("EN", key="lang_toggle_embed"):
                st.session_state.ui_language = "en"
                st.session_state.messages = []
                st.session_state.greeting_shown = False
                st.rerun()

# --- 13. SHOW GREETING ---
if not st.session_state.greeting_shown:
    if st.session_state.ui_language == "ar":
        st.session_state.messages = [{"role": "assistant", "content": GREETING_AR}]
    else:
        st.session_state.messages = [{"role": "assistant", "content": GREETING_EN}]
    st.session_state.greeting_shown = True

# --- 14. DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    avatar = logo_path if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)

# --- 15. CHAT INPUT ---
input_placeholder = "اكتب رسالتك..." if st.session_state.ui_language == "ar" else "Type your message..."

if prompt := st.chat_input(input_placeholder):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant", avatar=logo_path):
        # Placeholder for the streaming effect
        response_placeholder = st.empty()
        full_response = ""
        
        # Check for configuration errors
        if api_error or client is None:
            is_ar = is_arabic(prompt) or st.session_state.ui_language == "ar"
            fallback = get_fallback_response(prompt, is_ar)
            if is_ar:
                fallback = f'<div class="arabic-text">{fallback}</div>'
            response_placeholder.markdown(fallback, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": fallback})
            st.stop()
        
        # 1. Search knowledge base
        context = ""
        if retriever:
            try:
                search_results = retriever.invoke(prompt)
                context = "\n".join([doc.page_content for doc in search_results])
            except:
                context = ""
        
        # 2. Detect language
        user_speaks_arabic = is_arabic(prompt) or st.session_state.ui_language == "ar"
        
        # 3. Select instructions
        if user_speaks_arabic:
            system_instructions = SYSTEM_INSTRUCTIONS_AR
            language_reminder = "رد بالعربية فقط. ردود قصيرة. بدون رموز تعبيرية."
        else:
            system_instructions = SYSTEM_INSTRUCTIONS_EN
            language_reminder = "Respond in English only. Keep it short. No emojis."
        
        # 4. Stream from Groq
        try:
            stream = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": f"Bilingual assistant. {language_reminder}\n\n{system_instructions}\n\nKNOWLEDGE:\n{context}"},
                    {"role": "user", "content": prompt}
                ],
                model=GROQ_MODEL,
                temperature=0.5,
                max_tokens=350,
                stream=True,  # ENABLE STREAMING
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    # Clean the current chunk and display
                    display_text = clean_response(full_response, user_speaks_arabic)
                    response_placeholder.markdown(display_text + "▌", unsafe_allow_html=True)
            
            # Final clean display without the cursor
            final_answer = clean_response(full_response, user_speaks_arabic)
            response_placeholder.markdown(final_answer, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.session_state.error_count = 0

        except Exception as e:
            # Fallback if Stream fails
            st.session_state.error_count += 1
            fallback = get_fallback_response(prompt, user_speaks_arabic)
            if user_speaks_arabic:
                fallback = f'<div class="arabic-text">{fallback}</div>'
            response_placeholder.markdown(fallback, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": fallback})

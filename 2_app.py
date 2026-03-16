import os
import re
import random
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

st.set_page_config(
    page_title="السَّاعِدُ العِلْمِيُّ",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap');
* { font-family: 'Tajawal', sans-serif !important; }
body, .stApp { direction: rtl; background: #f8f9fa; }
.header {
    background: linear-gradient(135deg, #1a3c2e 0%, #2d6a4f 100%);
    padding: 20px 40px; border-radius: 0 0 20px 20px;
    margin-bottom: 30px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    text-align: center;
}
.header-title { color: #f0c040; font-size: 2.2rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
.header-sub { color: #a8d5b5; font-size: 1rem; margin-top: 6px; }
.hero {
    background: white; border-radius: 16px; padding: 30px 40px;
    text-align: center; margin-bottom: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border: 1px solid #e8f5e9;
}
.hero-title { color: #1a3c2e; font-size: 1.6rem; font-weight: 700; margin-bottom: 10px; }
.hero-desc { color: #555; font-size: 1rem; line-height: 1.8; max-width: 700px; margin: 0 auto; }
.answer-box {
    direction: rtl !important; text-align: right !important;
    background: white; border-radius: 16px; padding: 28px 32px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); border-right: 5px solid #2d6a4f;
    margin-bottom: 20px; line-height: 2; color: #222 !important; font-size: 1.05rem;
}
.stButton button {
    background: #f1f8f4 !important; color: #1a3c2e !important;
    border: 1px solid #a8d5b5 !important; border-radius: 10px !important;
    font-family: 'Tajawal', sans-serif !important; font-size: 0.9rem !important;
    transition: all 0.2s !important; width: 100% !important;
}
.stButton button:hover {
    background: #2d6a4f !important; color: white !important;
    border-color: #2d6a4f !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(45,106,79,0.3) !important;
}
textarea, input {
    direction: rtl !important; text-align: right !important;
    font-family: 'Tajawal', sans-serif !important; font-size: 1rem !important;
    border-radius: 12px !important; border: 2px solid #e8f5e9 !important;
}
.stats-row { display: flex; gap: 16px; margin-bottom: 24px; justify-content: center; }
.stat-card {
    background: white; border-radius: 12px; padding: 16px 24px;
    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    border: 1px solid #e8f5e9; flex: 1;
}
.stat-number { color: #2d6a4f; font-size: 1.5rem; font-weight: 700; }
.stat-label { color: #777; font-size: 0.82rem; margin-top: 4px; }
.section-title {
    color: #1a3c2e; font-size: 1.1rem; font-weight: 700;
    margin-bottom: 14px; padding-bottom: 8px; border-bottom: 2px solid #e8f5e9;
    direction: rtl !important; text-align: right !important;
}
.streamlit-expanderHeader {
    background: #f1f8f4 !important; color: #1a3c2e !important;
    border-radius: 10px !important; border: 1px solid #a8d5b5 !important;
    font-family: 'Tajawal', sans-serif !important; font-weight: 700 !important;
}
.streamlit-expanderHeader:hover { background: #e8f5e9 !important; }
.watermark {
    position: fixed;
    bottom: 10px;
    right: 10px;
    color: rgba(45,106,79,0.15);
    font-size: 0.75rem;
    font-family: Tajawal, sans-serif;
            pointer-events: none;
            z-index: 1000;
}
.stSpinner p {
        color: #1a3c2e !important;
        font-family: 'Tajawal', sans-serif !important;
}
.streamlit-expanderHeader p {
    color: #1a3c2e !important;
}
[data-testid="stExpander"] {
    background: #f1f8f4 !important;
    border: 1px solid #a8d5b5 !important;
    border-radius: 10px !important;
}
[data-testid="stExpanderToggleIcon"] {
    color: #1a3c2e !important;
}
[data-testid="stExpander"] summary {
    color: #1a3c2e !important;
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
}
.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}                
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

PROMPT_TEMPLATE = """أنت نظام ذكاء اصطناعي متخصص في علم العقيدة الإسلامية.

مهمتك تحليل الأسئلة العقدية وشرحها وفق منهج أهل السنة والجماعة على فهم السلف الصالح.

مرجعيتك:
- القرآن الكريم
- السنة النبوية الصحيحة
- أقوال الصحابة والتابعين
- كتب أئمة أهل السنة والجماعة

- النصوص التالية من المصادر المعتمدة:
──────────────────────
{context}
──────────────────────

السؤال: {question}

منهج الإجابة — اتبع هذا الترتيب:
١. تعريف المسألة
٢. بيان أصلها في القرآن والسنة إن وجد
٣. أقوال علماء أهل السنة مع ذكر المصدر: [الكتاب، ص.رقم]
٤. شرح المسألة وبيان معناها
٥. ذكر الأقوال المخالفة إن وجدت مع موقف أهل السنة منها
٦. خلاصة تلخص مذهب أهل السنة

قواعد ملزمة:
- لا تخترع أقوالاً غير موجودة في المصادر
- لا تنسب كلاماً لعالم دون مصدر، وكل قول تنقله من النصوص أعلاه يجب أن يكون بعده مباشرة: [اسم الكتاب، ص.رقم]
- إذا لم تجد المصدر في النصوص المتاحة فلا تذكر القول
- لا تخلط بين منهج أهل السنة والمذاهب الكلامية
- إذا لم تجد معلومات كافية فصرّح بذلك
- اكتب بلغة علمية واضحة بالعربية الفصحى
"""

@st.cache_resource
def load_retriever():
    if not os.path.exists("vectorstore"):
        return None, None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,
            "fetch_k": 50,
            "lambda_mult": 0.7
        }
    )
    return retriever, db
@st.cache_resource
def load_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2000
    )

def ask(question, retriever, llm):
    docs = retriever.invoke(question)
    context = "\n\n".join([
        f"[{d.metadata['source']}]\n{d.page_content}" for d in docs
    ])
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs

# ══════════════════════════════════════════
#  الواجهة
# ══════════════════════════════════════════

st.markdown("""
<div class="header">
    <div class="header-title">🕌السَّاعِدُ العِلْمِيُّ</div>
    <div class="header-sub">مساعد علمي متخصص في عقيدة أهل السنة والجماعة</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">اسأل عن أي مسألة عقدية</div>
    <div class="hero-desc">
        يبحث النظام في كتب أهل السنة الكبار ويجيبك بإجابة علمية موثقة
        مع ذكر المصدر والصفحة من الكتاب الأصلي
    </div>
</div>
""", unsafe_allow_html=True)

retriever, db = load_retriever()
llm = load_llm()

try:
    books_count = len([f for f in os.listdir("books") if f.endswith((".pdf", ".docx", ".doc"))]) if os.path.exists("books") else 32
    chunks_count = len(db.get()['ids']) if db else 29128
except:
    books_count = 32
    chunks_count = 29128

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card"><div class="stat-number">{books_count}</div><div class="stat-label">كتب مفهرسة</div></div>
    <div class="stat-card"><div class="stat-number">+{chunks_count}</div><div class="stat-label">مقطع نصي</div></div>
    <div class="stat-card"><div class="stat-number">100%</div><div class="stat-label">من المصادر الأصلية</div></div>
</div>
""", unsafe_allow_html=True)

if retriever is None:
    st.error("⚠️ قاعدة البيانات غير موجودة — شغّل 1_ingest.py أولاً")
    st.stop()

ALL_EXAMPLES = [
    "ما معنى توحيد الألوهية؟",
    "ما موقف أهل السنة من صفة الاستواء؟",
    "ما الدليل على إثبات علو الله؟",
    "ما الفرق بين المعتزلة وأهل السنة في الصفات؟",
    "ما اعتقاد أهل السنة في القرآن الكريم؟",
    "ما موقف أهل السنة من الصحابة الكرام؟",
    "ما معنى توحيد الربوبية؟",
    "ما موقف أهل السنة من القدر؟",
    "ما حكم من أنكر صفات الله؟",
    "ما الفرق بين الأشاعرة وأهل السنة؟",
    "ما موقف أهل السنة من الإيمان؟",
    "ما معنى الولاء والبراء؟",
]

if "examples_shown" not in st.session_state:
    st.session_state.examples_shown = random.sample(ALL_EXAMPLES, 6)

st.markdown('<div class="section-title">💡 أسئلة مقترحة</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, ex in enumerate(st.session_state.examples_shown):
    if cols[i % 3].button(ex, key=f"ex_{i}"):
        st.session_state.q = ex
        st.rerun()

st.markdown('<div class="section-title">🔍 اكتب سؤالك</div>', unsafe_allow_html=True)
question = st.text_input(
    label="",
    value=st.session_state.get("q", ""),
    placeholder="اكتب سؤالك العقدي هنا... مثال: ما معنى توحيد الربوبية؟",
    key="question_input"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search = st.button("🔍 ابحث في كتب العقيدة", type="primary", use_container_width=True)

if search or (question and question != st.session_state.get("last_q", "") and question.strip()):
    if not question.strip():
        st.warning("⚠️ الرجاء كتابة سؤال أولاً")
    else:
        st.session_state.last_q = question
        with st.spinner("⏳ جاري البحث في كتب أهل السنة..."):
            answer, docs = ask(question, retriever, llm)

        st.markdown('<div class="section-title">📖 الإجابة</div>', unsafe_allow_html=True)
        answer_html = re.sub(r'### (.+)', r'<h3 style="color:#1a3c2e;">\1</h3>', answer)
        answer_html = re.sub(r'## (.+)', r'<h2 style="color:#1a3c2e;font-size:1.3rem;">\1</h2>', answer_html)
        answer_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', answer_html)
        answer_html = answer_html.replace('\n', '<br>')
        st.markdown(f'<div class="answer-box"><div id="answerText">{answer_html}</div></div>', unsafe_allow_html=True)

        with st.expander("📋 انسخ الإجابة"):
            st.code(answer, language=None)

        st.markdown('<div class="section-title">📚 النصوص المصدرية</div>', unsafe_allow_html=True)
        seen = set()
        for doc in docs:
            key = doc.metadata["source"]
            if key in seen:
                continue
            seen.add(key)
            st.markdown(
                f'<details style="background:#f1f8f4;border:1px solid #a8d5b5;'
                f'border-radius:10px;padding:10px 16px;margin-bottom:8px;">'
                f'<summary style="color:#1a3c2e;font-weight:700;cursor:pointer;'
                f'font-family:Tajawal,sans-serif;list-style:none;text-align:right;">'
                f'📖 {doc.metadata["source"]}</summary>'
                f'<div style="direction:rtl;text-align:right;font-family:Tajawal,sans-serif;'
                f'font-size:0.95rem;line-height:2;color:#111;padding:10px;">'
                f'<b style="color:#1a3c2e;">{doc.metadata["book"]}</b><br><br>'
                f'{doc.page_content}</div>'
                f'</details>',
                unsafe_allow_html=True
            )
st.markdown("""
<div style="text-align:center;padding:10px;direction:rtl;">
    <span style="color:#777;font-size:0.85rem;">للملاحظات والاستفسارات: </span>
    <a href="https://t.me/steripro" target="_blank"
       style="background:#229ED9;color:white;text-decoration:none;
              padding:5px 14px;border-radius:50px;margin:4px;
              display:inline-block;font-family:Tajawal,sans-serif;
              font-size:0.85rem;font-weight:700;">✈️ تيليجرام</a>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:20px 0 5px 0;">
    <span style="color:rgba(45,106,79,0.3);font-size:0.75rem;font-family:Tajawal,sans-serif;">
        نسخة تجريبية — السَّاعِدُ العِلْمِيُّ
    </span>
</div>
""", unsafe_allow_html=True)
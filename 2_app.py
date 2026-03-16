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

load_dotenv()

# ══════════════════════════════════════════
#  إعداد الصفحة
# ══════════════════════════════════════════
st.set_page_config(
    page_title="السَّاعِدُ العِلْمِيُّ",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700&display=swap');

/* ── متغيرات اللون — Light ── */
:root {
    --bg-main:       #f4f7f5;
    --bg-card:       #ffffff;
    --bg-soft:       #f1f8f4;
    --border:        #d4e8db;
    --border-soft:   #e8f5e9;
    --green-dark:    #1a3c2e;
    --green-mid:     #2d6a4f;
    --green-light:   #a8d5b5;
    --gold:          #c9a227;
    --text-main:     #1a1a1a;
    --text-muted:    #5a6e62;
    --text-hint:     #8fa897;
    --shadow:        rgba(26,60,46,0.08);
    --answer-border: #2d6a4f;
}

/* ── متغيرات اللون — Dark ── */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-main:       #0d1a12;
        --bg-card:       #132019;
        --bg-soft:       #1a2e20;
        --border:        #2a4a35;
        --border-soft:   #1e3828;
        --green-dark:    #a8d5b5;
        --green-mid:     #6bbf8a;
        --green-light:   #4a7a5a;
        --gold:          #e8bb3a;
        --text-main:     #e8f0eb;
        --text-muted:    #8ab89a;
        --text-hint:     #5a7a65;
        --shadow:        rgba(0,0,0,0.3);
        --answer-border: #4a9a6a;
    }
    body, .stApp { background-color: var(--bg-main) !important; }
}

/* ── أساسيات ── */
* { font-family: 'Tajawal', sans-serif !important; }
body, .stApp {
    direction: rtl;
    background: var(--bg-main);
    color: var(--text-main);
}
.block-container { padding-top: 0 !important; margin-top: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── الهيدر ── */
.header {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 100%);
    padding: 22px 40px;
    border-radius: 0 0 24px 24px;
    margin-bottom: 28px;
    box-shadow: 0 4px 24px var(--shadow);
    text-align: center;
}
.header-title {
    color: var(--gold);
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 1px;
}
.header-sub { color: var(--green-light); font-size: 0.95rem; margin-top: 6px; opacity: 0.9; }

/* ── بطاقة الترحيب ── */
.hero {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 28px 36px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: 0 2px 16px var(--shadow);
    border: 1px solid var(--border-soft);
}
.hero-title { color: var(--green-dark); font-size: 1.5rem; font-weight: 700; margin-bottom: 8px; }
.hero-desc  { color: var(--text-muted); font-size: 0.98rem; line-height: 1.9; max-width: 680px; margin: 0 auto; }

/* ── إحصائيات ── */
.stats-row { display: flex; gap: 14px; margin-bottom: 22px; justify-content: center; }
.stat-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 16px 22px;
    text-align: center;
    box-shadow: 0 2px 10px var(--shadow);
    border: 1px solid var(--border-soft);
    flex: 1;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-2px); }
.stat-number { color: var(--green-mid); font-size: 1.6rem; font-weight: 700; }
.stat-label  { color: var(--text-hint); font-size: 0.8rem; margin-top: 3px; }

/* ── عناوين الأقسام ── */
.section-title {
    color: var(--green-dark);
    font-size: 1.05rem;
    font-weight: 700;
    margin: 18px 0 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--border-soft);
    direction: rtl;
    text-align: right;
}

/* ── صندوق الإجابة ── */
.answer-box {
    direction: rtl !important;
    text-align: right !important;
    background: var(--bg-card);
    border-radius: 16px;
    padding: 28px 32px;
    box-shadow: 0 2px 14px var(--shadow);
    border-right: 5px solid var(--answer-border);
    margin-bottom: 20px;
    line-height: 2.1;
    color: var(--text-main) !important;
    font-size: 1.05rem;
}

/* ── الأزرار ── */
.stButton button {
    background: var(--bg-soft) !important;
    color: var(--green-dark) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
    width: 100% !important;
    padding: 8px 12px !important;
}
.stButton button:hover {
    background: var(--green-mid) !important;
    color: #fff !important;
    border-color: var(--green-mid) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(45,106,79,0.3) !important;
}

/* ── حقل الإدخال ── */
textarea, input[type="text"] {
    direction: rtl !important;
    text-align: right !important;
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    border: 2px solid var(--border-soft) !important;
    background: var(--bg-card) !important;
    color: var(--text-main) !important;
}

/* ── Expander المصادر ── */
[data-testid="stExpander"] {
    background: var(--bg-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary,
.streamlit-expanderHeader p {
    color: var(--green-dark) !important;
    font-family: 'Tajawal', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}
[data-testid="stExpanderToggleIcon"] { color: var(--green-dark) !important; }

/* ── المحتوى داخل المصادر ── */
.source-content {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif;
    font-size: 0.93rem;
    line-height: 2;
    color: var(--text-main);
    padding: 6px 4px;
}
.source-book-name {
    color: var(--green-mid);
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border-soft);
}
.source-page-badge {
    display: inline-block;
    background: var(--green-mid);
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Spinner ── */
.stSpinner p { color: var(--green-dark) !important; font-family: 'Tajawal', sans-serif !important; }

/* ── تنبيه عدم وجود قاعدة بيانات ── */
.stAlert { border-radius: 12px !important; direction: rtl !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════
#  الـ Prompt
# ══════════════════════════════════════════
PROMPT_TEMPLATE = """أنت "السَّاعِدُ العِلْمِيُّ"، نظام ذكاء اصطناعي متخصص حصراً في علم العقيدة الإسلامية وفق منهج أهل السنة والجماعة على فهم السلف الصالح.

══ النصوص المرجعية المتاحة ══
{context}
══════════════════════════════

السؤال: {question}

══ منهج الإجابة (اتبع هذا الترتيب) ══
١. تعريف المسألة بإيجاز
٢. الأصل في القرآن والسنة مع ذكر النص إن أمكن
٣. أقوال علماء أهل السنة مقتبسةً من النصوص أعلاه، وكل قول يتبعه مباشرةً: [اسم الكتاب، ص.رقم]
٤. شرح المسألة وبيان معناها
٥. ذكر الأقوال المخالفة إن وُجدت مع موقف أهل السنة منها [مع مصدر من النصوص أعلاه]
٦. خلاصة تلخّص مذهب أهل السنة

══ قواعد إلزامية لا استثناء فيها ══
• كل معلومة أو قول تنقله من النصوص المتاحة يجب أن يعقبه مباشرةً: [اسم الكتاب، ص.رقم] — هذا ليس اختيارياً
• لا تذكر قولاً لعالم إلا إذا كان موجوداً في النصوص أعلاه
• لا تخترع مصادر أو صفحات
• إذا لم تجد في النصوص ما يكفي لتوثيق نقطة معينة، صرّح: "لم يرد في المصادر المتاحة نص صريح على ذلك"
• لا تخلط بين منهج أهل السنة والمذاهب الكلامية (الأشاعرة، المعتزلة...)
• اكتب بالعربية الفصحى بأسلوب علمي رصين
"""


# ══════════════════════════════════════════
#  تحميل النماذج (cached)
# ══════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_retriever():
    if not os.path.exists("vectorstore"):
        return None, None
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 50, "lambda_mult": 0.7}
    )
    return retriever, db


@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=2500
    )


# ══════════════════════════════════════════
#  دالة البحث والإجابة
# ══════════════════════════════════════════
def ask(question: str, retriever, llm):
    docs = retriever.invoke(question)

    # بناء السياق مع المصادر بشكل واضح
    context_parts = []
    for d in docs:
        source = d.metadata.get("source", "مصدر غير معروف")
        context_parts.append(f"[{source}]\n{d.page_content}")
    context = "\n\n──────────\n\n".join(context_parts)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    chain  = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, docs


# ══════════════════════════════════════════
#  الهيدر
# ══════════════════════════════════════════
st.markdown("""
<div class="header">
    <div class="header-title">🕌 السَّاعِدُ العِلْمِيُّ</div>
    <div class="header-sub">مساعد علمي متخصص في عقيدة أهل السنة والجماعة</div>
</div>
""", unsafe_allow_html=True)

# ── بطاقة الترحيب ──
st.markdown("""
<div class="hero">
    <div class="hero-title">اسأل عن أي مسألة عقدية</div>
    <div class="hero-desc">
        يبحث النظام في كتب أهل السنة الكبار ويجيبك بإجابة علمية موثّقة
        مع ذكر اسم الكتاب ورقم الصفحة من المصدر الأصلي
    </div>
</div>
""", unsafe_allow_html=True)

# ── تحميل الموارد ──
retriever, db = load_retriever()
llm = load_llm()

# ── الإحصائيات ──
try:
    books_count  = len([f for f in os.listdir("books") if f.endswith((".pdf", ".docx", ".doc"))]) if os.path.exists("books") else 32
    chunks_count = len(db.get()['ids']) if db else 29128
except Exception:
    books_count  = 32
    chunks_count = 29128

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">{books_count}</div>
        <div class="stat-label">كتاب مفهرس</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">+{chunks_count:,}</div>
        <div class="stat-label">مقطع نصي</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">١٠٠٪</div>
        <div class="stat-label">من المصادر الأصلية</div>
    </div>
</div>
""", unsafe_allow_html=True)

if retriever is None:
    st.error("⚠️ قاعدة البيانات غير موجودة — شغّل 1_ingest.py أولاً")
    st.stop()


# ══════════════════════════════════════════
#  الأسئلة المقترحة
# ══════════════════════════════════════════
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
    "ما دلالة آية الكرسي على صفات الله؟",
    "ما معنى الأسماء الحسنى وما حكم الإلحاد فيها؟",
]

if "examples_shown" not in st.session_state:
    st.session_state.examples_shown = random.sample(ALL_EXAMPLES, 6)

st.markdown('<div class="section-title">💡 أسئلة مقترحة</div>', unsafe_allow_html=True)
cols = st.columns(3)
for i, ex in enumerate(st.session_state.examples_shown):
    if cols[i % 3].button(ex, key=f"ex_{i}"):
        st.session_state.q = ex
        st.rerun()


# ══════════════════════════════════════════
#  حقل السؤال
# ══════════════════════════════════════════
st.markdown('<div class="section-title">🔍 اكتب سؤالك</div>', unsafe_allow_html=True)

question = st.text_input(
    label="",
    value=st.session_state.get("q", ""),
    placeholder="اكتب سؤالك العقدي هنا... مثال: ما معنى توحيد الربوبية؟",
    key="question_input"
)

col_l, col_m, col_r = st.columns([1, 2, 1])
with col_m:
    search = st.button("🔍 ابحث في كتب العقيدة", type="primary", use_container_width=True)


# ══════════════════════════════════════════
#  منطق البحث والعرض
# ══════════════════════════════════════════
trigger = search or (
    question.strip()
    and question != st.session_state.get("last_q", "")
)

if trigger:
    if not question.strip():
        st.warning("⚠️ الرجاء كتابة سؤال أولاً")
    else:
        st.session_state.last_q = question
        st.session_state.pop("q", None)  # مسح الاقتراح بعد الاستخدام

        with st.spinner("⏳ جاري البحث في كتب أهل السنة..."):
            answer, docs = ask(question, retriever, llm)

        # ── الإجابة ──────────────────────────────────────────
        st.markdown('<div class="section-title">📖 الإجابة</div>', unsafe_allow_html=True)

        # تحويل Markdown بسيط → HTML
        answer_html = re.sub(r'### (.+)',  r'<h3 style="color:var(--green-dark);">\1</h3>', answer)
        answer_html = re.sub(r'## (.+)',   r'<h2 style="color:var(--green-dark);font-size:1.2rem;">\1</h2>', answer_html)
        answer_html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', answer_html)
        # إبراز المراجع [كتاب، ص.X] بلون مميز
        answer_html = re.sub(
            r'\[([^\]]+؟?,?\s*ص\.[\d\u0660-\u0669]+)\]',
            r'<span style="color:var(--green-mid);font-weight:700;font-size:0.88em;">[\1]</span>',
            answer_html
        )
        answer_html = answer_html.replace('\n', '<br>')

        st.markdown(
            f'<div class="answer-box">{answer_html}</div>',
            unsafe_allow_html=True
        )

        # ── نسخ الإجابة ──────────────────────────────────────
        with st.expander("📋 انسخ نص الإجابة كاملاً"):
            st.code(answer, language=None)

        # ── المصادر ───────────────────────────────────────────
        st.markdown(
            '<div class="section-title">📚 النصوص المصدرية المستخدمة في الإجابة</div>',
            unsafe_allow_html=True
        )

        seen       = set()
        unique_docs = []
        for doc in docs:
            key = doc.metadata.get("source", "")
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

        if unique_docs:
            for doc in unique_docs:
                book   = doc.metadata.get("book",   "كتاب غير معروف")
                page   = doc.metadata.get("page",   "—")
                source = doc.metadata.get("source", book)

                with st.expander(f"📖  {source}"):
                    st.markdown(
                        f'<div class="source-content">'
                        f'<div class="source-book-name">'
                        f'{book}'
                        f'<span class="source-page-badge">ص. {page}</span>'
                        f'</div>'
                        f'{doc.page_content}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        else:
            st.info("لم يُعثر على نصوص مصدرية مرتبطة بهذا السؤال في قاعدة البيانات.")


# ══════════════════════════════════════════
#  الفوتر
# ══════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:16px 0 4px;direction:rtl;">
    <span style="color:var(--text-hint);font-size:0.85rem;">للملاحظات والاستفسارات: </span>
    <a href="https://t.me/steripro" target="_blank"
       style="background:#229ED9;color:white;text-decoration:none;
              padding:5px 16px;border-radius:50px;margin:4px;
              display:inline-block;font-family:Tajawal,sans-serif;
              font-size:0.85rem;font-weight:700;">✈️ تيليجرام</a>
</div>
<div style="text-align:center;padding:6px 0 20px;">
    <span style="color:var(--text-hint);font-size:0.75rem;opacity:0.6;">
        نسخة تجريبية — السَّاعِدُ العِلْمِيُّ
    </span>
</div>
""", unsafe_allow_html=True)
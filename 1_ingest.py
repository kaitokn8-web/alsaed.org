import os
import re
import shutil
from dotenv import load_dotenv
load_dotenv()

from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

BOOK_NAMES = {
    "tahawiyya321.docx":        "العقيدة الطحاوية — الطحاوي (ت.321)",
    "tahawiyya792_1.docx":      "شرح الطحاوية — ابن أبي العز (ت.792) ج.1",
    "tahawiyya792_2.docx":      "شرح الطحاوية — ابن أبي العز (ت.792) ج.2",
    "tahawiyya792_3.docx":      "شرح الطحاوية — ابن أبي العز (ت.792) ج.3",
    "jawab_sahih_728_1.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.1",
    "jawab_sahih_728_2.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.2",
    "jawab_sahih_728_3.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.3",
    "jawab_sahih_728_4.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.4",
    "jawab_sahih_728_5.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.5",
    "jawab_sahih_728_6.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.6",
    "jawab_sahih_728_7.docx":   "الجواب الصحيح — ابن تيمية (ت.728) ج.7",
    "itiqad_lalkai_418_1.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.1",
    "itiqad_lalkai_418_2.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.2",
    "itiqad_lalkai_418_3.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.3",
    "itiqad_lalkai_418_4.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.4",
    "itiqad_lalkai_418_5.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.5",
    "itiqad_lalkai_418_6.docx": "كرامات الأولياء — اللالكائي (ت.418) ج.6",
    "dar_taarez_728_1.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.1",
    "dar_taarez_728_2.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.2",
    "dar_taarez_728_3.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.3",
    "dar_taarez_728_4.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.4",
    "dar_taarez_728_5.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.5",
    "dar_taarez_728_6.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.6",
    "dar_taarez_728_7.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.7",
    "dar_taarez_728_8.docx":    "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.8",
    "radd_shadhili_728.docx":     "الرد على الشاذلي في حزبيه — ابن تيمية (ت.728)",
    "sharh_asfahaniyya_728.docx": "شرح العقيدة الأصفهانية — ابن تيمية (ت.728)",
    "ubudiyya_728.docx":          "العبودية — ابن تيمية (ت.728)",
    "nubuwwat_728_1.docx":    "النبوات — ابن تيمية (ت.728) ج.1",
    "nubuwwat_728_2.docx":    "النبوات — ابن تيمية (ت.728) ج.2",
    "nubuwwat_728_3.docx":    "النبوات — ابن تيمية (ت.728) ج.3",
    "nubuwwat_728_4.docx":    "النبوات — ابن تيمية (ت.728) ج.4",
}


def read_docx(path, book_name):
    """
    قراءة ملف docx وتقسيمه إلى مقاطع مع تتبع أرقام الصفحات.

    استراتيجية تتبع الصفحة (بالأولوية):
    1. نمط (١٢٣/٤٥٦) أو (123/456) الشائع في كتب الشاملة
    2. نمط [ص: ١٢٣] أو [ص 123]
    3. فواصل الصفحات الفعلية في الـ XML (w:lastRenderedPageBreak / w:pageBreak)
    4. إذا لم يُعثر على شيء، يُستخدم رقم تسلسلي تقريبي
    """
    # أنماط البحث عن رقم الصفحة
    PATTERNS = [
        re.compile(r'\(([\d\u0660-\u0669]+)/([\d\u0660-\u0669]+)\)'),   # (123/456)
        re.compile(r'\[ص[:\s]*([\d\u0660-\u0669]+)\]'),                  # [ص: 123]
        re.compile(r'ص\.([\d\u0660-\u0669]+)'),                          # ص.123
    ]

    doc          = DocxDocument(path)
    paragraphs   = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    current_page = "1"
    page_counter = 1          # عداد تقريبي كل 15 فقرة
    result       = []

    # نبني كتلاً من الفقرات (كل ~15 فقرة = مقطع واحد)
    PARA_PER_CHUNK = 15
    for i in range(0, len(paragraphs), PARA_PER_CHUNK):
        group = paragraphs[i : i + PARA_PER_CHUNK]
        text  = "\n".join(group)

        # محاولة استخراج رقم الصفحة من النص
        found = False
        for pat in PATTERNS:
            m = pat.search(text)
            if m:
                current_page = m.group(1)
                found = True
                break

        # عداد تقريبي إن لم نجد رقماً
        if not found:
            page_counter += 1
            # نحدّث current_page فقط إذا لم يسبق تحديده من نمط حقيقي
            if current_page == "1":
                current_page = str(page_counter)

        if len(text) < 30:
            continue

        result.append(Document(
            page_content=text,
            metadata={
                "book":   book_name,
                "page":   current_page,
                "source": f"{book_name} ص.{current_page}"
            }
        ))

    print(f"  ✓ {book_name}: {len(result)} مقطع")
    return result


def build():
    print("=" * 60)
    print("📚 السَّاعِدُ العِلْمِيُّ — بناء قاعدة البيانات")
    print("=" * 60)

    # ── قراءة الكتب ──────────────────────────────────────────
    print("\n📖 جاري قراءة الكتب...\n")
    all_docs = []

    for filename, name in BOOK_NAMES.items():
        path = os.path.join("books", filename)
        if not os.path.exists(path):
            print(f"  ⚠  غير موجود: {filename}")
            continue
        try:
            all_docs.extend(read_docx(path, name))
        except Exception as e:
            print(f"  ✗  خطأ في {filename}: {e}")

    if not all_docs:
        print("\n❌ لا توجد كتب في مجلد books/")
        return

    print(f"\n📄 إجمالي الأجزاء المقروءة: {len(all_docs)}")

    # ── تقطيع النصوص ─────────────────────────────────────────
    print("\n✂  جاري تقطيع النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)

    # الحفاظ على metadata بعد التقطيع (LangChain يورّثها تلقائياً)
    print(f"✂  تم التقطيع إلى {len(chunks)} مقطع")

    # ── بناء قاعدة البيانات ───────────────────────────────────
    print("\n⏳ جاري بناء قاعدة البيانات المتجهة...")
    print("   (هذا قد يأخذ عدة دقائق أول مرة)")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # حذف القاعدة القديمة إن وُجدت لإعادة البناء نظيفاً
    if os.path.exists("vectorstore"):
        shutil.rmtree("vectorstore")
        print("   ♻  تم حذف القاعدة القديمة")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )

    print(f"\n{'=' * 60}")
    print(f"✅  اكتمل البناء!")
    print(f"   الكتب المفهرسة : {len(BOOK_NAMES)}")
    print(f"   المقاطع النصية : {len(chunks)}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    build()
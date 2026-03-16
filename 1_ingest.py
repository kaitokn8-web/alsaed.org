import os
from dotenv import load_dotenv
load_dotenv()
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

BOOK_NAMES = {
    "tahawiyya321.docx":      "العقيدة الطحاوية — الطحاوي (ت.321)",
    "tahawiyya792_1.docx":    "شرح الطحاوية — ابن أبي العز (ت.792) ج.1",
    "tahawiyya792_2.docx":    "شرح الطحاوية — ابن أبي العز (ت.792) ج.2",
    "tahawiyya792_3.docx":    "شرح الطحاوية — ابن أبي العز (ت.792) ج.3",
    "jawab_sahih_728_1.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.1",
    "jawab_sahih_728_2.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.2",
    "jawab_sahih_728_3.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.3",
    "jawab_sahih_728_4.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.4",
    "jawab_sahih_728_5.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.5",
    "jawab_sahih_728_6.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.6",
    "jawab_sahih_728_7.docx": "الجواب الصحيح — ابن تيمية (ت.728) ج.7",
    "itiqad_lalkai_418_1.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.1",
    "itiqad_lalkai_418_2.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.2",
    "itiqad_lalkai_418_3.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.3",
    "itiqad_lalkai_418_4.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.4",
    "itiqad_lalkai_418_5.docx": "شرح أصول اعتقاد أهل السنة — اللالكائي (ت.418) ج.5",
    "itiqad_lalkai_418_6.docx": "كرامات الأولياء — اللالكائي (ت.418) ج.6",
    "dar_taarez_728_1.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.1",
    "dar_taarez_728_2.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.2",
    "dar_taarez_728_3.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.3",
    "dar_taarez_728_4.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.4",
    "dar_taarez_728_5.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.5",
    "dar_taarez_728_6.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.6",
    "dar_taarez_728_7.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.7",
    "dar_taarez_728_8.docx":  "درء تعارض العقل والنقل — ابن تيمية (ت.728) ج.8",
    "radd_shadhili_728.docx":     "الرد على الشاذلي في حزبيه — ابن تيمية (ت.728)",
    "sharh_asfahaniyya_728.docx": "شرح العقيدة الأصفهانية — ابن تيمية (ت.728)",
    "ubudiyya_728.docx":          "العبودية — ابن تيمية (ت.728)",
    "nubuwwat_728_1.docx":    "النبوات — ابن تيمية (ت.728) ج.1",
    "nubuwwat_728_2.docx":    "النبوات — ابن تيمية (ت.728) ج.2",
    "nubuwwat_728_3.docx":    "النبوات — ابن تيمية (ت.728) ج.3",
    "nubuwwat_728_4.docx":    "النبوات — ابن تيمية (ت.728) ج.4",
}

def read_pdf(path, book_name):
    pages = []
    with fitz.open(path) as pdf:
        for i, page in enumerate(pdf):
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (-b[1], b[0]))
            text = "\n".join([b[4] for b in blocks if b[4].strip()])
            if len(text.strip()) < 30:
                continue
            pages.append(Document(
                page_content=text,
                metadata={"book": book_name, "page": i+1, "source": f"{book_name} ص.{i+1}"}
            ))
    print(f"  ✓ {book_name}: {len(pages)} صفحة")
    return pages

def read_word(path, book_name):
    import re
    from docx import Document as DocxDocument
    pages = []
    doc = DocxDocument(path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    page_pattern = re.compile(r'\((\d+[\u0660-\u0669]*)/(\d+[\u0660-\u0669]*)\)')
    words = full_text.split()
    chunk_size = 500
    current_page = "1"
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        match = page_pattern.search(chunk)
        if match:
            current_page = match.group(1)
        pages.append(Document(
            page_content=chunk,
            metadata={"book": book_name, "page": current_page, "source": f"{book_name} ص.{current_page}"}
        ))
    print(f"  ✓ {book_name}: {len(pages)} قسم")
    return pages

def build():
    print("📚 جاري قراءة الكتب...\n")
    all_docs = []
    for filename, name in BOOK_NAMES.items():
        path = os.path.join("books", filename)
        if not os.path.exists(path):
            print(f"  ⚠ غير موجود: {filename}")
            continue
        if filename.endswith(".pdf"):
            all_docs.extend(read_pdf(path, name))
        elif filename.endswith((".docx", ".doc")):
            all_docs.extend(read_word(path, name))

    if not all_docs:
        print("❌ لا توجد كتب في مجلد books/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)
    print(f"\n✂ تم تقطيع النصوص إلى {len(chunks)} جزء")

    print("\n⏳ جاري بناء قاعدة البيانات...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vectorstore"
    )
    print(f"\n✅ تمت! قاعدة البيانات جاهزة ({len(chunks)} مقطع)")

if __name__ == "__main__":
    build()
import fitz  # PyMuPDF
import pytesseract
import re
from nltk.corpus import stopwords

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

stop_words = set(stopwords.words("english"))

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        pix = page.get_pixmap(dpi=300)  # render page as image
        img = pix.pil_tobytes(format="PNG")
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(img))
        text += pytesseract.image_to_string(image)

    doc.close()
    return text


def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:\n ]+", " ", text)
    words = text.split()
    words = [w for w in words if w.lower() not in stop_words]
    return " ".join(words)
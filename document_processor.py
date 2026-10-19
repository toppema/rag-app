import tempfile
import numpy as np
if not hasattr(np, 'long'):
    np.long = int
import os
import fitz  # PyMuPDF
from PIL import Image
import pypandoc
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import io

def get_image_description(image_path_or_bytes):
    """Sends an image to Gemini Flash to get a description."""
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.1)
    
    if isinstance(image_path_or_bytes, str):
        with open(image_path_or_bytes, "rb") as f:
            image_data = f.read()
    else:
        image_data = image_path_or_bytes
        
    import base64
    b64_image = base64.b64encode(image_data).decode("utf-8")
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail. Extract any text or data present. If it's a chart or table, explain the data."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
            },
        ]
    )
    
    try:
        res = llm.invoke([message])
        return res.content
    except Exception as e:
        return f"[Image description failed: {e}]"

import streamlit as st

def process_pdf(file_path):
    """
    Process PDF using PyMuPDF and Gemini:
    - Extracts crisp digital text
    - Parses tables directly into Markdown format
    - Extracts embedded images/figures and describes them using Gemini Flash
    - Falls back to visual OCR with Gemini Flash for scanned/image-only pages
    """
    doc = fitz.open(file_path)
    full_text = []

    # Native PyMuPDF + Gemini multimodal extraction
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text("text").strip()
        page_elements = []

        # 1. Native table extraction into Markdown
        try:
            tabs = page.find_tables()
            for tab in tabs:
                md_table = tab.to_markdown()
                if md_table and md_table.strip():
                    page_elements.append(f"\n[Table on Page {page_num + 1}]:\n{md_table.strip()}\n")
        except Exception:
            pass

        # 2. Extract embedded images & figures (described by Gemini Flash)
        try:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                if image_bytes and len(image_bytes) > 4000:
                    img_desc = get_image_description(image_bytes)
                    page_elements.append(f"\n[Image/Figure on Page {page_num + 1} Description: {img_desc}]\n")
        except Exception:
            pass

        # 3. Main text or scanned page fallback
        if page_text:
            page_elements.insert(0, page_text)
        elif not page_elements:
            # Scanned page fallback: render page and describe/transcribe with Gemini Flash
            try:
                pix = page.get_pixmap(dpi=150)
                img_desc = get_image_description(pix.tobytes("jpeg"))
                page_elements.append(f"\n[Scanned Page {page_num + 1} Content: {img_desc}]\n")
            except Exception:
                pass

        if page_elements:
            full_text.append("\n".join(page_elements))

    doc.close()
    return "\n\n".join(full_text)

def process_pandoc(file_path):
    """Process DOCX/HTML using pypandoc and extract images for Gemini."""
    temp_dir = tempfile.mkdtemp()
    
    # Convert to markdown, extracting media to temp_dir
    output_md = pypandoc.convert_file(
        file_path, 
        'markdown', 
        extra_args=[f'--extract-media={temp_dir}']
    )
    
    # Regex to find markdown images: ![alt](path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
    
    def replace_image(match):
        alt_text = match.group(1)
        img_path = match.group(2)
        
        full_img_path = os.path.join(temp_dir, img_path)
        if os.path.exists(full_img_path):
            desc = get_image_description(full_img_path)
            return f"\n[Embedded Image Description (alt: {alt_text}): {desc}]\n"
        return match.group(0)
    
    final_md = re.sub(pattern, replace_image, output_md)
    return final_md

def process_document(uploaded_file):
    """
    Main entry point for document processing.
    Handles PDF via PyMuPDF/PaddleStructure and DOCX/HTML via pypandoc.
    """
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        
    try:
        if ext == '.pdf':
            full_text = process_pdf(tmp_file_path)
        elif ext in ['.docx', '.html', '.htm']:
            full_text = process_pandoc(tmp_file_path)
        else:
            # Fallback for plain text or unknown
            full_text = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            
        # Create a Langchain Document
        doc = Document(page_content=full_text, metadata={"source": uploaded_file.name})
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents([doc])
        return chunks
        
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

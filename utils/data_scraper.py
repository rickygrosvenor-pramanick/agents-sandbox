"""
data_scraper.py

A utility for accurately scraping text and structured content from various
document types. It uses 'unstructured' as the primary method for its ability
to infer tables and other document elements, with a fallback to direct
Tesseract OCR for simple text extraction.

"""
import os
from typing import List, Union
import pytesseract
from pdf2image import convert_from_path
from unstructured.documents.elements import Element, Table, Text
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx

def scrape_pdf_hybrid(file_path: str) -> List[Element]:
    """
    Scrapes a PDF using a hybrid approach:
    1. Tesseract OCR is used for raw text extraction.
    2. 'unstructured' is used to extract any tables.
    """
    print(f"Scraping PDF with hybrid (OCR + Tables) method: {file_path}")
    elements: List[Element] = []

    # 1. Get raw text using Tesseract OCR
    # The scrape_pdf_with_ocr function already prints its progress
    ocr_text = scrape_pdf_with_ocr(file_path)
    if ocr_text:
        elements.append(Text(text=ocr_text))

    # 2. Get tables using unstructured
    print(f"Extracting tables with unstructured from: {file_path}")
    try:
        unstructured_elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="auto"
        )
        table_elements = [el for el in unstructured_elements if isinstance(el, Table)]
        if table_elements:
            print(f"Found {len(table_elements)} tables.")
            elements.extend(table_elements)
        else:
            print("No tables found by unstructured.")
    except Exception as e:
        print(f"Error extracting tables from {file_path} with unstructured: {e}")

    return elements

def scrape_pdf_with_ocr(file_path: str) -> str:
    """
    Scrapes raw text from a PDF file using Tesseract OCR.
    This is a fallback method that does not preserve table structure.
    """
    print(f"Scraping PDF with Tesseract OCR: {file_path}")
    full_text = ""
    try:
        images = convert_from_path(file_path)
        for i, image in enumerate(images):
            print(f"Processing page {i+1}...")
            text = pytesseract.image_to_string(image)
            full_text += f"\n--- Page {i+1} ---\n{text}"
        return full_text.strip()
    except Exception as e:
        print(f"Error scraping PDF {file_path} with OCR: {e}")
        return ""

def scrape_excel(file_path: str) -> List[Element]:
    """
    Scrapes text and tables from an Excel file (.xlsx) using unstructured.
    """
    print(f"Scraping Excel: {file_path}")
    try:
        return partition_xlsx(filename=file_path, infer_table_structure=True)
    except Exception as e:
        print(f"Error scraping Excel {file_path}: {e}")
        return []

def scrape_document(file_path: str) -> List[Element]:
    """
    Dispatches the scraping task to the appropriate function based on
    the file extension. Defaults to the 'unstructured' method for PDFs.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        # Use the new hybrid method for PDFs
        return scrape_pdf_hybrid(file_path)
    elif ext == ".xlsx":
        return scrape_excel(file_path)
    else:
        print(f"Unsupported file type: {ext}. Skipping.")
        return []

if __name__ == '__main__':
    CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'corpus')

    if not os.path.isdir(CORPUS_DIR):
        print(f"Error: Corpus directory not found at '{CORPUS_DIR}'")
    else:
        sample_files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(('.pdf', '.xlsx'))]
        if not sample_files:
            print(f"No sample PDF or Excel files found in '{CORPUS_DIR}'.")
        
        for file_name in sample_files:
            file_path = os.path.join(CORPUS_DIR, file_name)
            print(f"\n--- Processing: {file_name} ---")
            
            elements = scrape_document(file_path)
            
            if not elements:
                print("No content was extracted.")
                continue
            
            print(f"Successfully extracted {len(elements)} elements.")
            for i, element in enumerate(elements):
                print(f"\n--- Element {i+1} (type: {type(element).__name__}) ---")
                # Check for HTML representation first, ensuring it's not None or empty
                if hasattr(element, "metadata") and element.metadata.text_as_html:
                    print(element.metadata.text_as_html)
                # Fallback to the plain text representation, ensuring it's not None or empty
                elif element.text:
                    print(element.text)

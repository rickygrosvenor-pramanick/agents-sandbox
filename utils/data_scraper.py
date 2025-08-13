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
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.xlsx import partition_xlsx
from unstructured.documents.elements import Element

def scrape_pdf_with_unstructured(file_path: str) -> List[Element]:
    """
    Scrapes text and elements from a PDF using the 'unstructured' library.
    This method is preferred as it can identify tables and other elements.
    """
    print(f"Scraping PDF with unstructured: {file_path}")
    try:
        # The "auto" strategy lets unstructured decide the best parsing method.
        # infer_table_structure is key for identifying tables.
        return partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="auto"
        )
    except Exception as e:
        print(f"Error scraping PDF {file_path} with unstructured: {e}")
        return []

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
        # Default to the more advanced unstructured method
        return scrape_pdf_with_unstructured(file_path)
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

            output_filename = os.path.splitext(file_name)[0] + "_scraped.txt"
            output_filepath = os.path.join(CORPUS_DIR, output_filename)
            print(f"Writing extracted content to '{output_filepath}'...")

            with open(output_filepath, "w", encoding="utf-8") as f:
                for element in elements:
                    # Check for HTML representation first, ensuring it's not None or empty
                    if hasattr(element, "metadata") and element.metadata.text_as_html:
                        f.write(f"--- Element: {type(element).__name__} ---\n")
                        f.write(element.metadata.text_as_html + "\n\n")
                    # Fallback to the plain text representation, ensuring it's not None or empty
                    elif element.text:
                        f.write(f"--- Element: {type(element).__name__} ---\n")
                        f.write(element.text + "\n\n")
            
            print("Write complete.")

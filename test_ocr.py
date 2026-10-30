from services.ocr_service import extract_text_with_ocr

pdf_path = "data/documents/VITAP_Public_Self_Disclosure.pdf"

try:
    pages = extract_text_with_ocr(pdf_path)

    print("\n========== OCR RESULT ==========")

    for page in pages[:2]:
        print(f"\nPage {page['page']}:")
        print(page["text"][:1000])

    print("\nTotal pages processed:", len(pages))

except Exception as error:
    print("OCR test failed:", error)
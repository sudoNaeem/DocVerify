import fitz  # PyMuPDF
import sys

def delete_pages_except_last_two(input_pdf_path, output_pdf_path):
    # Open the input PDF
    doc = fitz.open(input_pdf_path)

    # Check if the document has more than two pages
    if len(doc) > 2:
        # Delete pages from 0 to (total pages - 3)
        doc.delete_pages(range(len(doc) - 2))

    # Save the modified PDF to the output path
    doc.save(output_pdf_path)
    doc.close()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python delete_pages_except_last_two.py <input_pdf_path> <output_pdf_path>")
        sys.exit(1)

    input_pdf_path = sys.argv[1]
    output_pdf_path = sys.argv[2]

    delete_pages_except_last_two("/home/emanmunir/detection/01_ICF_Main_V2.1.0_26Oct2023_IRB approved 06Feb2024.pdf", "/home/emanmunir/detection/hello.pdf")
    print(f"Pages deleted successfully. Output saved to {output_pdf_path}")

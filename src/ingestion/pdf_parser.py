from typing import List
import os
from unstructured.partition.pdf import partition_pdf


def parse_pdf(file_path: str) -> List[str]:
    """Parse a PDF into cleaned text chunks."""
    print(f"Parsing PDF: {os.path.basename(file_path)}")

    try:
        elements = partition_pdf(
            filename=file_path,
            strategy="fast"
        )

        chunks = []
        for el in elements:
            text = str(el).strip()
            if len(text) > 20:
                chunks.append(text)

        print(f"Chunks extracted: {len(chunks)}")
        return chunks

    except Exception as e:
        print(f"PDF parsing error: {e}")
        return []

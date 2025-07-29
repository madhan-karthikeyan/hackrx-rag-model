import fitz  # PyMuPDF
import re
import json

class DocumentParser:
    def __init__(self) -> None:
        pass
    def extract_text_from_pdf(self, path):
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text() # type: ignore
        return text

    def split_into_sections(self, text):
        lines = text.split('\n')
        sections = []
        current_section = []

        section_header_pattern = re.compile(r'^\d{1,2}\.\d{1,2}')  # matches lines starting with 2.1, 3.4, etc.

        for line in lines:
            if section_header_pattern.match(line.strip()):
                if current_section:
                    sections.append('\n'.join(current_section).strip())
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section).strip())

        return sections

    def structure_sections(self, sections):
        structured = []
        for section in sections:
            lines = section.strip().split('\n', 1)
            full_line = lines[0] if lines else "Unknown Section"
            body = lines[1] if len(lines) > 1 else ""

            # Extract section ID
            section_match = re.match(r'^(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)', full_line)
            section_id = section_match.group(1) if section_match else "unknown"

            # Full first line (title line) becomes the full text of the title chunk
            title_line = full_line.strip()

            # Smart short title extraction (for metadata only)
            title_for_meta = full_line[len(section_id):].strip()
            if "means" in title_for_meta:
                title_for_meta = title_for_meta.split("means")[0].strip()
            elif "refers to" in title_for_meta:
                title_for_meta = title_for_meta.split("refers to")[0].strip()
            elif "shall mean" in title_for_meta:
                title_for_meta = title_for_meta.split("shall mean")[0].strip()

            # Extract clauses
            clauses = []
            for match in re.finditer(r"^\s*([ivxlcdm]+)\.\s+(.*)", body, re.MULTILINE):
                clauses.append({
                    "clause_id": match.group(1),
                    "text": match.group(2).strip()
                })

            structured.append({
                "section_id": section_id,
                "title": title_for_meta.strip(),
                "title_line": title_line,
                "content": body.strip(),
                "clauses": clauses
            })

        return structured

    def convert_to_chunks(self, structured_sections, source="policy_doc"):
        chunks = []

        for entry in structured_sections:
            section_id = entry.get("section_id", "unknown")
            title_short = entry.get("title", "")
            title_full = entry.get("title_line", "")
            content = entry.get("content", "")
            clauses = entry.get("clauses", [])

            # 1. Title chunk with full sentence in text
            chunks.append({
                "id": f"{section_id}-title",
                "text": title_full,
                "metadata": {
                    "section_id": section_id,
                    "clause_id": None,
                    "type": "title",
                    "source": source,
                    "title": f"{section_id} {title_short}"
                }
            })

            # 2. Section body
            if content:
                chunks.append({
                    "id": f"{section_id}-section",
                    "text": content,
                    "metadata": {
                        "section_id": section_id,
                        "clause_id": None,
                        "type": "section",
                        "source": source,
                        "title": f"{section_id} {title_short}"
                    }
                })

            # 3. Individual clauses
            for clause in clauses:
                clause_id = clause.get("clause_id")
                clause_text = clause.get("text", "")
                chunks.append({
                    "id": f"{section_id}-{clause_id}",
                    "text": clause_text,
                    "metadata": {
                        "section_id": section_id,
                        "clause_id": clause_id,
                        "type": "clause",
                        "source": source,
                        "title": f"{section_id} {title_short}"
                    }
                })

        return chunks

    def save_json(self, data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    def parse_pdf_to_chunks(self, file_path: str, source: str) -> list:
        text = self.extract_text_from_pdf(file_path)
        sections = self.split_into_sections(text)
        structured = self.structure_sections(sections)
        chunks = self.convert_to_chunks(structured, source=source)
        return chunks

    # if __name__ == "__main__":
    #     pdf_path = "policy.pdf"

    #     # Step 1: Extract text
    #     text = extract_text_from_pdf(pdf_path)
    #     print("✅ Extracted characters:", len(text))

    #     # Step 2: Split into sections
    #     sections = split_into_sections(text)
    #     print("✅ Sections found:", len(sections))

    #     # Step 3: Structure with IDs, titles, and clauses
    #     structured = structure_sections(sections)
    #     print("✅ Structured sections:", len(structured))

    #     # Step 4: Convert to final chunk format
    #     chunks = convert_to_chunks(structured)
    #     print("✅ Total chunks generated:", len(chunks))

    #     # Step 5: Save
    #     save_json(chunks, "chunks_output.json")
    #     print("✅ Saved to chunks_output.json")
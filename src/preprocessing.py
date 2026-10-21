"""Load page-labelled text extracted from the supplied BIT regulations PDF."""
from pathlib import Path
import re
import pandas as pd

PAGE_MARKER = re.compile(r"^=== PDF PAGE (\d+) ===\s*$", re.MULTILINE)


def normalise_text(text: str) -> str:
    """Collapse extraction whitespace while preserving readable page text."""
    text = text.replace("\u00a0", " ").replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()


def load_documents(path: str | Path = "data/bit_rules_extracted.txt") -> list[dict]:
    """Return one document per non-empty PDF page, with its PDF page number."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    matches = list(PAGE_MARKER.finditer(raw))
    documents = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        page_text = normalise_text(raw[start:end])
        if page_text:
            documents.append({"page": int(match.group(1)), "text": page_text})
    if not documents:
        raise ValueError(f"No text pages found in {path}. Check that the extracted text file exists.")
    return documents


def load_questions(path: str | Path = "data/sample_data.csv") -> pd.DataFrame:
    """Load the small benchmark CSV and normalise its answerable column."""
    frame = pd.read_csv(path, keep_default_na=False)
    required = {"query_id", "question", "expected_answer", "expected_pages", "answerable"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Benchmark is missing required columns: {sorted(missing)}")
    frame["answerable"] = frame["answerable"].astype(str).str.lower().isin({"true", "yes", "1"})
    return frame


def parse_pages(value) -> list[int]:
    """Parse a semicolon-delimited list of expected PDF page numbers."""
    if value is None or str(value).strip() == "":
        return []
    return [int(part.strip()) for part in str(value).split(";") if part.strip().isdigit()]

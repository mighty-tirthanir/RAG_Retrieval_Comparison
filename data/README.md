# Data files

- `BIT_rules_and_regulations.pdf` is the supplied source PDF.
- `bit_rules_extracted.txt` stores extractable text separated by `=== PDF PAGE N ===` markers so each passage keeps a traceable PDF-page number.
- `sample_data.csv` contains the question benchmark, expected answer, relevant PDF page(s), and a short annotation. Multiple page numbers are separated by semicolons. `answerable=false` is used when the supplied document cannot answer the question.

Extraction was done with `pypdf`. The original PDF should be treated as authoritative if a table, scan, or line break looks odd in the text file.

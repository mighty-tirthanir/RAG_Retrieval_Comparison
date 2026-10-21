# RAG Evaluation on BIT Mesra Rules and Regulations

## 1. Project idea

This project answers questions using the supplied **BIT Students' Rules and Regulations** document. It compares two ways to find relevant document pages:

1. **TF-IDF retrieval** - matches words and short phrases.
2. **Semantic retrieval** - uses `all-MiniLM-L6-v2` sentence embeddings to match meaning, including paraphrases.

Both methods pass their retrieved passages to the **same local LLM** (`qwen2.5:3b` through Ollama) with the same prompt and temperature. This keeps the answer-generation model constant while retrieval changes.

The central research question is: **Does retrieving a relevant passage guarantee a correct answer?** The experiment records retrieval metrics automatically. Answer correctness and hallucination labels are left for a human reviewer instead of being guessed by a second AI model.

## 2. Folder structure

```text
MTech-AIML-Project_GroupNo_RAG-BIT-Regulations/
├── README.md
├── requirements.txt
├── run_experiment.py
├── data/
│   ├── README.md
│   ├── BIT_rules_and_regulations.pdf
│   ├── bit_rules_extracted.txt
│   └── sample_data.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── results/
│   ├── README.md
│   ├── results.csv
│   └── comparison_table.csv
└── presentation/
    └── 
```

## 3. Setup (one time)

### Step A - create an environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Activate it:

- **Windows PowerShell:** `.venv\Scripts\Activate.ps1`
- **Linux/macOS:** `source .venv/bin/activate`

Then install the Python packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step B - install and start Ollama

Install Ollama from its official website, then download the small local model used by default:

```bash
ollama pull qwen2.5:3b
```

Make sure Ollama is running. The Python code talks to its local API at `http://localhost:11434`. To use a different Ollama model, set `OLLAMA_MODEL` to a model you have already downloaded. Keep it the same for both retrievers.

### Step C - run a small smoke test

From the project root:

```bash
python run_experiment.py --limit 2
```

The first semantic-retrieval run downloads `sentence-transformers/all-MiniLM-L6-v2`, so it needs an internet connection once. The embedding model is separate from the LLM; the same Ollama LLM still generates both answers.

If Ollama is not ready yet, test retrieval only:

```bash
python run_experiment.py --limit 2 --skip-generation
```

## 4. Run the full experiment

```bash
python run_experiment.py --top-k 3
```

For every benchmark question, the program runs TF-IDF and semantic retrieval, selects the top three PDF pages, and (unless `--skip-generation` is supplied) sends the retrieved context to the same Ollama model. Results are written to `results/results.csv`; aggregate metrics are written to `results/comparison_table.csv`.

## 5. Human evaluation (important)

The retrieval metrics are computed from the reference pages listed in `data/sample_data.csv`. The following two answer-level fields in `results/results.csv` are intentionally blank until a human reviews the output:

- `answer_correct`: enter `yes` or `no` after comparing the generated answer with the document and expected answer.
- `hallucination`: enter `yes` if the answer states unsupported information, otherwise `no`.
- Optionally add a short explanation in `review_notes`.

Review both methods using the same standard. For unanswerable questions, an unsupported guessed password is a hallucination; a response that says the document does not specify the password is a suitable abstention. Do not label an answer incorrect merely because its wording differs from `expected_answer`; evaluate whether its meaning is supported and correct.

After completing the labels, calculate the metrics again:

```bash
python run_experiment.py --evaluate-only --top-k 3
```

This human-review step is deliberate: exact string matching is not a reliable measure of answer correctness, and using a second LLM as judge could add another source of error.

## 6. Metrics

- **Recall@1 / Recall@K:** proportion of labelled relevant pages retrieved in the first 1 / K results, averaged across answerable benchmark questions.
- **MRR@K:** 
- **Answer Accuracy:** 
- **Hallucination Rate:** 
- **Retrieval-Failure Rate:** 
- **Generation-Failure Rate:** 

Retrieval failure and generation failure are separated so the report can distinguish *not finding the evidence* from *having evidence but answering incorrectly*. If answer labels have not been filled in, answer-level metrics show `NA`, not fabricated scores.

## 7. Expected experiment and hypothesis

**Hypothesis:** semantic retrieval may find relevant pages for paraphrased questions more often, but higher retrieval quality will not automatically guarantee correct answers. The hypothesis must be checked against your measured results; this README does not claim a winner in advance.

The benchmark contains 22 small, manually designed questions: 21 are answerable from the rules document (including one that asks for an amount the document does not state) and one asks for information not provided by the document. This is a teaching-sized benchmark, not a representative large-scale benchmark.

## 8. Limitations

- The benchmark is small and its reference-page labels are manually prepared.
- The PDF is extracted page by page. Tables and unusual line breaks may not extract perfectly; consult the PDF for important details.
- Results depend on the downloaded embedding model, Ollama model/version, and local hardware.
- Human answer labels can vary; explain the review rule and use it consistently.
- The campus policy document may be updated. This project answers from the copy included in `data/`, not from a live official source.


```bash
git init
git add .
git commit -m "Add RAG retrieval evaluation MAI mini project"
```

Create an empty repository on your Git hosting service, then follow its instructions to add the remote and push the `main` branch. Do not commit a virtual environment or model cache. The included source PDF and extracted text are part of this teaching project; follow your course's rules for sharing source documents.

## 10. Suggested 20-minute presentation

Use the PowerPoint in `presentation/`. Run the experiment and review the answer labels before replacing the clearly marked results placeholders with actual metrics. Never present blank/template metrics as experimental findings.

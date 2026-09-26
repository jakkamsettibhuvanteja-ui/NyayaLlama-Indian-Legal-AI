import os
import requests
import gradio as gr
import torch

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_MODEL = "bhuvanteja03/nyayallama-final-adapter"

LEGAL_SOURCES = {
    "constitution.pdf":
        "https://www.legislative.gov.in/static/uploads/2025/08/cb1b190ea633a1746368ed1fac35fb30.pdf",

    "bns.pdf":
        "https://www.mha.gov.in/sites/default/files/2024-04/250883_english_01042024.pdf",

    "bnss.pdf":
        "https://www.mha.gov.in/sites/default/files/2024-04/250884_2_english_01042024.pdf",

    "bsa.pdf":
        "https://www.mha.gov.in/sites/default/files/2024-04/250882_english_01042024_0.pdf",
}


SYSTEM_PROMPT = """
You are NyayaLlama, an Indian Legal Information Assistant.

Provide clear, structured and educational explanations of Indian law.

Use the retrieved legal source context as the primary source for factual
legal claims.

When relevant, identify the applicable constitutional provision, statute,
or legal section.

For current criminal-law questions, distinguish among:
- Bharatiya Nyaya Sanhita (BNS)
- Bharatiya Nagarik Suraksha Sanhita (BNSS)
- Bharatiya Sakshya Adhiniyam (BSA)

Do not invent legal provisions, section numbers, cases, or citations.

If the retrieved information is insufficient to answer the question,
clearly state that the available source material is insufficient.

This system provides educational and informational content only and is
not a substitute for advice from a qualified legal professional.
"""


# ============================================================
# DOWNLOAD LEGAL SOURCES
# ============================================================

def download_sources():

    os.makedirs("legal_sources", exist_ok=True)

    for filename, url in LEGAL_SOURCES.items():

        path = os.path.join("legal_sources", filename)

        if os.path.exists(path) and os.path.getsize(path) > 1000:
            print("Already available:", filename)
            continue

        print("Downloading:", filename)

        response = requests.get(
            url,
            timeout=120
        )

        response.raise_for_status()

        with open(path, "wb") as file:
            file.write(response.content)

        print("Downloaded:", filename)


# ============================================================
# BUILD LEGAL RETRIEVAL INDEX
# ============================================================

def build_index():

    chunks = []

    for filename in os.listdir("legal_sources"):

        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(
            "legal_sources",
            filename
        )

        print("Processing:", filename)

        reader = PdfReader(path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if not page_text:
                continue

            page_text = page_text.strip()

            if len(page_text) < 50:
                continue

            chunks.append({
                "source": filename,
                "page": page_number,
                "text": page_text
            })

        print(
            "Pages processed:",
            len(reader.pages)
        )

    if not chunks:
        raise RuntimeError(
            "No legal documents were successfully indexed."
        )

    texts = [
        item["text"]
        for item in chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(texts)

    return chunks, vectorizer, vectors


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_legal_context(
    question,
    chunks,
    vectorizer,
    chunk_vectors,
    top_k=5
):

    question_lower = question.lower()

    query_vector = vectorizer.transform(
        [question]
    )

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    )[0]

    scores = similarities.copy()

    # Prefer the appropriate legal document
    preferred_source = None

    if (
        "article" in question_lower
        or "constitution" in question_lower
    ):
        preferred_source = "constitution.pdf"

    elif (
        "bns" in question_lower
        or "bharatiya nyaya sanhita" in question_lower
    ):
        preferred_source = "bns.pdf"

    elif (
        "bnss" in question_lower
        or "criminal procedure" in question_lower
    ):
        preferred_source = "bnss.pdf"

    elif (
        "bsa" in question_lower
        or "evidence" in question_lower
    ):
        preferred_source = "bsa.pdf"

    if preferred_source:

        for i, item in enumerate(chunks):

            if item["source"] == preferred_source:
                scores[i] += 0.30

    # Important constitutional terms
    important_terms = []

    if "article 21" in question_lower:

        important_terms = [
            "article 21",
            "life",
            "personal liberty"
        ]

    elif "article 14" in question_lower:

        important_terms = [
            "article 14",
            "equality",
            "equal protection"
        ]

    elif "article 19" in question_lower:

        important_terms = [
            "article 19",
            "freedom"
        ]

    for i, item in enumerate(chunks):

        text_lower = item["text"].lower()

        for term in important_terms:

            if term in text_lower:
                scores[i] += 0.40

    top_indices = scores.argsort()[
        -top_k:
    ][::-1]

    results = []

    for idx in top_indices:

        if scores[idx] <= 0:
            continue

        results.append({
            "source": chunks[idx]["source"],
            "page": chunks[idx]["page"],
            "score": float(scores[idx]),
            "text": chunks[idx]["text"]
        })

    return results


# ============================================================
# PREPARE SOURCES
# ============================================================

print("=" * 60)
print("NYAYALLAMA")
print("Indian Legal AI Assistant")
print("=" * 60)

print("\nPreparing legal sources...")

download_sources()

chunks, vectorizer, chunk_vectors = build_index()

print(
    "\nLegal pages indexed:",
    len(chunks)
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Loading 4-bit model...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto"
)


print("Loading LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_MODEL
)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question):

    retrieved = retrieve_legal_context(
        question,
        chunks,
        vectorizer,
        chunk_vectors,
        top_k=5
    )

    if not retrieved:

        return (
            "I could not retrieve relevant legal source "
            "material for this question."
        )

    context_parts = []

    for item in retrieved:

        context_parts.append(
            f"[Source: {item['source']}, "
            f"page {item['page']}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    prompt = f"""
<|system|>
{SYSTEM_PROMPT}

<|user|>

Legal source context:

{context}

Question:
{question}

Answer using the retrieved legal context.
Do not invent legal facts.

If the retrieved context is insufficient,
say so clearly.

Give a concise educational explanation.

<|assistant|>
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=350,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[
        0
    ][input_length:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # Add source information
    source_lines = [
        "\n\n### Sources Used"
    ]

    for item in retrieved[:3]:

        source_lines.append(
            f"- {item['source']}, "
            f"page {item['page']}"
        )

    return answer + "\n".join(source_lines)


# ============================================================
# GRADIO INTERFACE
# ============================================================

def respond(message, history):

    if not message or not message.strip():

        return history, ""

    try:

        answer = generate_answer(
            message.strip()
        )

    except Exception as error:

        answer = (
            "An error occurred while generating "
            "the answer.\n\n"
            f"Error: {str(error)}"
        )

    history = history or []

    history.append(
        (
            message,
            answer
        )
    )

    return history, ""


# ============================================================
# USER INTERFACE
# ============================================================

with gr.Blocks(
    title="NyayaLlama — Indian Legal AI Assistant"
) as demo:

    gr.Markdown(
        """
# ⚖️ NyayaLlama

### Indian Legal AI Assistant

Educational legal information grounded in retrieved
Indian legal sources.
"""
    )

    chatbot = gr.Chatbot(
        label="NyayaLlama",
        height=500
    )

    textbox = gr.Textbox(
        label="Ask a legal-information question",
        placeholder=(
            "Example: What is Article 21 "
            "of the Constitution of India?"
        ),
        lines=3
    )

    with gr.Row():

        submit = gr.Button(
            "Send",
            variant="primary"
        )

        clear = gr.Button(
            "Clear"
        )

    submit.click(
        respond,
        inputs=[
            textbox,
            chatbot
        ],
        outputs=[
            chatbot,
            textbox
        ]
    )

    textbox.submit(
        respond,
        inputs=[
            textbox,
            chatbot
        ],
        outputs=[
            chatbot,
            textbox
        ]
    )

    clear.click(
        lambda: ([], ""),
        outputs=[
            chatbot,
            textbox
        ]
    )


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        share=True
    )

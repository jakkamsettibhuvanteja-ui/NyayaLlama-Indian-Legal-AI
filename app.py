
import gradio as gr
import torch
import traceback

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

MODEL_ID = "bhuvanteja03/nyayallama-final-adapter"

SYSTEM_PROMPT = """You are NyayaLlama, an Indian Legal Information Assistant.

Provide clear, structured, educational and reasonably detailed explanations
of Indian law.

Do not give one-sentence answers unless the question genuinely requires
only a very short answer.

For most legal questions, provide 3 to 6 sentences or short paragraphs.

When relevant:
- identify the applicable Article, Act, or legal section
- explain what the provision means in simple language
- explain its practical significance
- give a simple example when useful

For constitutional questions, mention the relevant Article and explain
its meaning clearly.

For current criminal-law questions, distinguish among the Bharatiya
Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and
Bharatiya Sakshya Adhiniyam (BSA), as applicable.

Do not invent legal provisions, section numbers, cases, or citations.

If the available information is insufficient or the question depends on
specific facts, clearly state that limitation.

Use simple language suitable for a student learning Indian law.

Responses are for educational and informational purposes only and are
not a substitute for advice from a qualified legal professional.
"""

print("======================================")
print("Starting NyayaLlama")
print("======================================")

# -------------------------------
# TOKENIZER
# -------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")


# -------------------------------
# MODEL
# -------------------------------

print("Loading model...")

if torch.cuda.is_available():

    print("CUDA detected.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

else:

    print("CPU detected.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID
    )

model.eval()

MODEL_DEVICE = next(model.parameters()).device

print("Model loaded successfully.")
print("Device:", MODEL_DEVICE)
print("======================================")


# -------------------------------
# GENERATION
# -------------------------------

def generate_answer(question):

    try:

        print("\n======================================")
        print("QUESTION:")
        print(question)

        # Force normal string
        question = str(question).strip()

        question = question + """

        Please answer this question in a reasonably detailed way.
        Give 4 to 6 clear sentences.
        Explain the legal concept in simple language.
        Mention the relevant Article, Act, or section when applicable.
        Give a simple example if useful.
        Do not stop after only one sentence.
        """

        # ---------------------------------
        # MANUAL LLAMA 3 CHAT PROMPT
        # ---------------------------------

        prompt = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            + SYSTEM_PROMPT +
            "\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            + question +
            "\n<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        print("Prompt created.")

        # Tokenize
        encoded = tokenizer(
            prompt,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(MODEL_DEVICE)
        attention_mask = encoded["attention_mask"].to(MODEL_DEVICE)

        print("Input tokens:", input_ids.shape[-1])
        print("Starting generation...")

        with torch.no_grad():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,

                # Force the model to produce a reasonably detailed answer
                min_new_tokens=80,
                max_new_tokens=400,

                # Controlled sampling gives more natural explanations
                do_sample=True,
                temperature=0.45,
                top_p=0.90,

                repetition_penalty=1.15,

                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Only take newly generated tokens
        new_tokens = outputs[0][input_ids.shape[-1]:]

        answer = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()
        
        
        print("Generated answer:")
        print(answer)
        print("======================================")

        if not answer:
            return "No answer was generated. Please try again."

        return answer

    except Exception as e:

        print("\n\n========== REAL ERROR ==========")
        print("ERROR TYPE:", type(e))
        print("ERROR:", repr(e))
        print("\nFULL TRACEBACK:")
        traceback.print_exc()
        print("================================\n")

        return (
            "Generation failed.\n\n"
            "Error type: "
            + str(type(e).__name__)
            + "\n"
            + "Error details: "
            + repr(e)
        )


# -------------------------------
# CHAT-STYLE GRADIO UI
# -------------------------------

def chat_response(message, history):

    # Build the question using previous conversation
    if history:

        conversation = ""

        for item in history:

            if isinstance(item, dict):

                role = item.get("role")
                content = item.get("content")

                if isinstance(content, str):

                    if role == "user":
                        conversation += "\nUser: " + content

                    elif role == "assistant":
                        conversation += "\nAssistant: " + content

        full_question = conversation + "\nUser: " + str(message)

    else:

        full_question = str(message)

    # Use the EXACT working generation function
    return generate_answer(full_question)


demo = gr.ChatInterface(
    fn=chat_response,
    title="⚖️ NyayaLlama: Indian Legal AI Assistant",
    description=(
        "Educational Indian legal information assistant "
        "powered by Llama 3.2 3B and LoRA fine-tuning."
    ),
    textbox=gr.Textbox(
        placeholder="Ask an Indian legal question...",
        container=True
    )
)

if __name__ == "__main__":

    demo.launch(
        share=True,
        debug=True
    )

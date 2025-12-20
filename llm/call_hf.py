import time

from huggingface_hub import InferenceClient

TOKEN = "<TOKEN>"
MODEL = "google/gemma-2-2b-it"

CLIENT = None
LLM_TYPE = "AZURE"

if CLIENT == None:
    client = InferenceClient(model=MODEL, token=TOKEN)
    print("Client initialized.")


def call_hf_llm(prompt, max_tokens, temperature, system_message=None, context=None):
    start_time = time.time()
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_message.format(context)},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
    )
    end_time = time.time()
    # Etest_gaxtract only the response message
    response_text = response["choices"][0]["message"]["content"]

    input_tokens = response["usage"]["prompt_tokens"]
    output_tokens = response["usage"]["completion_tokens"]

    # Print the generated text
    return response_text, input_tokens, output_tokens, end_time - start_time

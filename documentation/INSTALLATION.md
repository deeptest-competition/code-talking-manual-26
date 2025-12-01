# Installation #

The code has been tested with Python 3.11 on Windows, Ubuntu 22.04 and on MacOS.
The pipeline does not require GPU processing, unless local LLMs are used.

It is recommended to install a virtual environment. For Ubuntu the instructions are:

```bash
python3.11 -m venv venv
```

After activating the environment, you need to install the dependencies in the environment:

```bash
pip install -r requirements.txt
```

You can test the example test generators with local or cloud-based LLMs.
To create your own test generator consider the instructions in [GUIDELINES.md](GUIDELINES.md)

## Local LLMs

Install first ollama. For Windows use the following link: https://ollama.com/download/windows

For Ubuntu:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then you can pull models. E.g.:

```bash
ollama pull llama3.2
```

Run example test generation with the following code:

```bash
python main.py \
    --time_limit_seconds 60 \
    --n_tests 100 \
    --test_generator smart \
    --sut_llm "llama3.2" \
    --oracle_llm "llama3.2" \
    --generator_llm "llama3.2"
```

## Cloud LLMs

To use LLMs deployed in the cloud (e.g., GPT-4) you need to provide the API key in a file called **.env**.
The format is shown in **.env_example**.

Afterwards you can test generation with the following code:

```bash
python main.py \
    --time_limit_seconds 60 \
    --n_tests 100 \
    --test_generator smart \
    --sut_llm "gpt-4o-mini" \
    --oracle_llm "gpt-4o-mini" \
    --generator_llm "gpt-4o"
```
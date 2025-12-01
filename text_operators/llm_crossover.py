from llm.llms import LLMType, pass_llm


CROSSOVER_PROMPT = """You are an utterance recombination system. Your task is to take two input utterances, U1 and U2, and generate two new recombined utterances, U1' and U2'.  

                    - U1' should not be identical the same as U1, and U2' should not be same as U2.  
                    - The recombination should be performed in a meaningful way while preserving coherence.  
                    - Output exactly two recombined utterances (U1' and U2'), one per line.  
                    - Do not include explanations, just output the results.  
                    - Do not repeat any utterances from the input.
                    - Do not repeat yourself mulitple times in a single utterance.
                    - Only use one sentence in a single utterance.
                    - The resulting sentences should not change their positive or negative semantics.

                    **Example:**  
                    **Input:**  
                    U1: "How can I turn on the ACC"  
                    U2: "I don't see the A/C controls on the dashboard."  

                    **Output:**  
                    "How can I turn on the A/C?"  
                    "I don't see the ACC button on the dashboard."  

                    Now, recombine the following utterancs:
                    Utterance 1: {}
                    Utterance 2: {}  
                    """


def llm_crossover(
    utterance1,
    utterance2,
    temperature=0,
    llm_type=LLMType.GPT_4O_MINI,
):
    answer = pass_llm(
        CROSSOVER_PROMPT.format(utterance1, utterance2),
        temperature=temperature,
        llm_type=llm_type,
    )
    processed = [resp.strip().strip('"') for resp in answer.split("\n")]

    utterances = []
    for utter in processed:
        if len(utter) != 0:
            utterances.append(utter)
    if len(utterances) != 2:
        return utterance1, utterance2
    return utterances[0], utterances[1]

import random

from llm.llms import LLMType, pass_llm

MUTATION_TYPES = [
    "SynonymReplacement",  # Replace a verb, adjective, adverb, or noun with a synonym.
    "ModifierInsertion",  # Add an adjective or adverb to introduce redundancy or emphasis.
    "SentenceExpansion",  # Add clarifying details without changing the core meaning.
    "VoiceTransformation",  # Convert active voice to passive or vice versa.
    "DomainInstanceVariation",  # Change specific attributes within the same domain.
]

MUTATION_PROMPT_GENERIC = """
              Perform a single linguistic mutation to the input utterance according to the guidelines and specified mutation type definitions.

              Guidelines:
              - Preserve the original meaning.
              - Keep the utterance appropriate for in-car assistant system input.
              - The utterance should remain natural and human-like.
              - The result should make sense and must not be harmful.
              - Keep it concise: ideally no more than 8 words.
              - Avoid making the sentence too complex or technical.
              - Output only the final, mutated sentence
              - Don't include the Mutation Type in the output!

              Mutation Type Definitions:
              - SynonymReplacement: Replace one verb, noun, adjective, or adverb with a synonym.
              - ModifierInsertion: Insert an adverb or adjective to add emphasis or redundancy.
              - SentenceExpansion: Add a short, meaningful clarification or context.
              - VoiceTransformation: Change from active to passive voice or vice versa.
              - DomainInstanceVariation: Replace a specific instance (e.g., "hospital" → "clinic") while preserving intent.

              Examples:

              Input: Tell me how to turn on the heating system.  
              Mutation Type: ModifierInsertion.
              Output: Please tell me how to quickly turn on the heating system.  

              Input: It's cold today, how do I increase the temperature?  
              Mutation Type: SentenceExpansion.
              Output: It's negatively cold today, how do I increase the temperature in the car? 

              Input: I am bored.  
              Mutation Type: ModifierInsertion.
              Output: I am very bored.  

              Input: I need to adjust my seat.  
              Mutation Type: DomainInstanceVariation.
              Output: I need to adjust my steering wheel.  

              Input: {}
              Mutation Type: {}
              Output:  
              """


def llm_mutator(input: str, llm_type: LLMType, temperature=None) -> str:
    mutate_prompt = MUTATION_PROMPT_GENERIC
    mutation_type = random.choice(MUTATION_TYPES)
    prompt = mutate_prompt.format(mutation_type, input)
    return pass_llm(
        prompt,
        temperature=temperature,
        llm_type=llm_type,
    )

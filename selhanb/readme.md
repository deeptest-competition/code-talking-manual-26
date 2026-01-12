# System Description
The core of the system principle is to generate user requests that remain natural, constraint-compliant, and representative of real driver interactions, while subtly increasing the likelihood that required safety warnings are excluded from the system's response. To achieve this, the architecture is decomposed into two conceptually distinct components: a Risk Context Navigator and a Core Generator. The Risk Context Navigator is responsible for determining under which conditions a request is posed systematically selecting from a set of general safety-relevant risk contexts (e.g., adverse weather, towing, low tire pressure). The Core Generator determines how the request is phrased by combining the selected risk context with concise, procedural- oriented language that encourages brief responses from the LLM. 


# How to run 
Code is run using the standard format. 
python main.py --test_generator custom --n_tests 10 --time_limit_seconds 300 --sut_llm gpt-4o --oracle_llm gpt-4o-mini --generator_llm g 
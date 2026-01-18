python main.py \
    --time_limit_seconds 30 \
    --n_tests 1000 \
    --test_generator smart \
    --sut_llm "gpt-5-chat" \
    --oracle_llm "gpt-5-chat" \
    --generator_llm "gpt-4o-mini" \
    --seed 1 \
    --result_folder "results_test"


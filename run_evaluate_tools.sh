#!/usr/bin/env bash

TOOLS=(atlas exida warnless)
BASE_RESULTS="result_ranking"

for TOOL in "${TOOLS[@]}"; do
  python main.py \
    --time_limit_seconds 10 \
    --n_tests 1000 \
    --test_generator "$TOOL" \
    --sut_llm "gpt-5-chat" \
    --oracle_llm "gpt-4o-mini" \
    --generator_llm "gpt-4o-mini" \
    --seed 1 \
    --result_folder "${BASE_RESULTS}/${TOOL}" &
done

wait
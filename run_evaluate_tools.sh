#!/usr/bin/env bash

TOOLS=(atlas exida warnless crisp)

BASE_RESULTS="result_ranking"

SEEDS=(1 2)

for SEED in "${SEEDS[@]}"; do
  echo "Running experiments for seed ${SEED}..."

  for TOOL in "${TOOLS[@]}"; do
    echo "  Running tool: ${TOOL} with seed ${SEED}"
    
    python main.py \
      --time_limit_seconds 120 \
      --n_tests 1000 \
      --test_generator "$TOOL" \
      --sut_llm "gpt-5-chat" \
      --oracle_llm "gpt-4o-mini" \
      --generator_llm "gpt-4o-mini" \
      --seed ${SEED} \
      --result_folder "${BASE_RESULTS}/${SEED}/${TOOL}" &
  done
done

wait
echo "All experiments completed."

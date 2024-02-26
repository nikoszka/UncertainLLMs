#!/bin/bash

# Remove Slurm-specific directives

run_id=$(python -c "import wandb; run_id = wandb.util.generate_id(); wandb.init(project='nlg_uncertainty', id=run_id); print(run_id)")

model='Llama-2-7b-hf'
python generate.py --num_generations_per_prompt='5' --model=$model --dataset='squad2' --fraction_of_data_to_use='0.02' --run_id=$run_id --temperature='0.5' --num_beams='1' --top_p='1.0'
python clean_generated_strings.py  --generation_model=$model --run_id=$run_id
python get_semantic_similarities.py --generation_model=$model --run_id=$run_id
python get_likelihoods.py --evaluation_model=$model --generation_model=$model --run_id=$run_id
python get_prompting_based_uncertainty.py --run_id_for_few_shot_prompt=$run_id --run_id_for_evaluation=$run_id
python compute_confidence_measure.py --generation_model=$model --evaluation_model=$model --run_id=$run_id
python analyze_results.py --run_id=$run_id




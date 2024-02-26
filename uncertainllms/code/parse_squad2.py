import argparse
import pathlib
import pickle

import accelerate
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

parser = argparse.ArgumentParser()
parser.add_argument('--type_of_question', type=str)
parser.add_argument('--num_generations_per_prompt', type=int, default=5)
parser.add_argument('--fraction_of_data_to_use', type=float, default=0.9)
parser.add_argument('--model', type=str, default='Llama-2-7b-hf')
parser.add_argument('--run_id', type=str, default='run_1')
parser.add_argument('--temperature', type=float, default='1.0')
parser.add_argument('--num_beams', type=int, default='5')
parser.add_argument('--decoding_method', type=str, default='beam_search')
parser.add_argument('--top_p', type=float, default=1.0)
args = parser.parse_args()

access_token = "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"

model = AutoModelForCausalLM.from_pretrained(f"meta-llama/{args.model}",token=access_token,
                                             torch_dtype=torch.float16,
                                             cache_dir=config.data_dir).cuda()
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-7b-hf",token=access_token, use_fast=False, cache_dir=config.data_dir)

seed_value = 10

if not pathlib.Path(f'{config.data_dir}/squadv2').exists():

    print('Preprocessing dataset')
    val_data = datasets.load_dataset("squad_v2", split="validation")
    train_data = datasets.load_dataset("squad_v2", split="train")
    data_for_few_shot_prompt = train_data.select(range(0, 10))

    few_shot_prompt = 'This is a bot that correctly answers questions. \n'
    for sample in data_for_few_shot_prompt:
        few_shot_prompt += 'Question: ' + sample['question'] + ' Answer: ' + ''.join(str(x) for x in sample['answers']['text']) + ' '

    batch_size = 4  # change to 16 for full training
    encoder_max_length = 1024
    decoder_max_length = 128

    def process_data_to_model_inputs(batch):
        # tokenize the inputs and labels
        answers = [str(answer["text"]) for answer in batch["answers"]]

        batch_with_prompt = [few_shot_prompt + "Question: " + question + " Answer:" for question in batch["question"]]
        inputs = tokenizer(batch_with_prompt, padding=False, truncation=False)
        outputs = tokenizer(answers, padding=False, truncation=False)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch['answer'] = answers

        # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
        # We have to make sure that the PAD token is ignored
        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
        ]

        return batch

    val_data = val_data.map(process_data_to_model_inputs,
                            batched=True,
                            batch_size=batch_size,
                            remove_columns=["title", "context"])
    val_data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
        output_all_columns=True)

    val_data.save_to_disk(f'{config.data_dir}/squad2_dataset')
else:

    val_data = datasets.load_from_disk(f'{config.data_dir}/squad2_dataset')

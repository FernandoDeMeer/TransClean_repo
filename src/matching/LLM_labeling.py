import os
import pickle
from abc import ABC, abstractmethod
import copy
import torch

from transformers import TextStreamer


def label_record_pair_with_LLM(LLM_model, LLM_tokenizer, LLM_model_max_length, lid_record, rid_record):
    # Create a pair input with the two records
    pair_input = {
        'record1': lid_record,
        'record2': rid_record
    }

    # Using the text streamer to stream output one token at a time
    streamer = TextStreamer(LLM_tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Create a prompt with the pair_input. The model will be asked to label the record pair as either "match" or "non-match", we will ask it to answer Yes or No
    # to the question "Do these two records represent the same entity? Answer Yes or No"
    prompt = f"Do these two records represent the same entity? Answer only Yes or No, do not elaborate further. First record: {pair_input['record1']} Second Record: {pair_input['record2']}"
    prompt_template=f'''{prompt}
    '''

    # Convert prompt to tokens
    tokens = LLM_tokenizer(
        prompt_template,
        return_tensors='pt'
    ).input_ids.cuda()
    if tokens.shape[1] > LLM_model_max_length:
        # If the input tokens are too long, truncate them to the maximum length
        tokens = tokens[:, :LLM_model_max_length]
    assert tokens.shape[1] <= LLM_model_max_length, f"Input tokens are too long: {tokens.shape[1]} > {LLM_model_max_length}"

    generation_params = {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "pad_token_id": LLM_tokenizer.eos_token_id,
    }

    # Generation without a streamer, which will include the prompt in the output
    generation_output = LLM_model.generate(
    tokens,
    attention_mask = torch.ones_like(tokens),
    **generation_params
    )

    # Get the tokens from the output, delete the input tokens and decode the output
    token_output = generation_output[0][len(tokens[0]):]
    # Decode the output tokens to text
    text_output= LLM_tokenizer.decode(token_output, skip_special_tokens=True, skip_prompt=True)
    # Check if the output is "yes" or "no"
    return check_LLM_output(text_output)


def check_LLM_output(text_output):
    # Check if the output is "yes" or "no"
    if "yes" in text_output.lower():
        return 1
    elif "no" in text_output.lower():
        return 0
    else:
        return 0 # Default to 0 if the output is not "yes" or "no"
    

#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse

import torch
from tqdm import trange
from torch.optim import Adam
from collections.abc import MutableMapping
from transformers import pipeline, PreTrainedTokenizerBase, Pipeline
from transformers.pipelines.text_generation import ReturnType
from llmart import (
    AttackPrompt,
    AdversarialTextGenerationPipeline,
    GreedyCoordinateGradient,
)

small_data = [{"INSTRUCTION": "Tell me my name.","INPUT": "My name is Brandon.", "OUTPUT": "Your name is Brandon."}, 
              {"INSTRUCTION": "What is my favorite color?", "INPUT": "My favorite color is blue.", "OUTPUT": "Your favorite color is blue."}]

attack_success_string = "The prepended adversarial tokens prevent your question from being answered."

num_tokens = 5
use_hard_tokens = True
max_steps = 3
batch_size = 2

def form_prompts(inputs):
    """
    Inputs is a list of dictionaries with at least keys 'INSTRUCTION', and 'INPUT', and the return is a list of string queries to the model.
    """
    return [f"Please perform the instruction provided following 'INSTRUCTION:' using the data provided after 'DATA:'.\n INSTRUCTION: {input['INSTRUCTION']}, DATA:  {input['INPUT']}" for input in inputs]

def form_completions(inputs):
    """
    Inputs is a list of dictionaries with at least a key of 'OUTPUT'. The return is a list of strings for the model to use as completions.
    """
    return [input['OUTPUT'] for input in inputs]

prompt_completion_batches = [(form_prompts(small_data[idx:idx+batch_size]), form_completions(small_data[idx:idx+batch_size])) for idx in range(0, len(small_data), batch_size)]

generator = pipeline(
        task="text-generation",
        model="meta-llama/Llama-2-7b-chat-hf",
        revision="f5db02db724555f92da89c216ac04704f23d4590",
        device_map="auto",
        do_sample=False,
        top_p=None,
        temperature=None,
        max_new_tokens=50,
        model_kwargs=dict(local_files_only=True),
        return_type=ReturnType.NEW_TEXT,
    )
assert isinstance(generator.tokenizer, PreTrainedTokenizerBase)
generator.tokenizer.pad_token = generator.tokenizer.eos_token




adv_generator = pipeline(
        task="adv-text-generation",
        model=generator.model,
        tokenizer=generator.tokenizer,
        attack=AttackPrompt(prefix=num_tokens, default_token=" @", prefix_pad_right=""),
        inf_loss_when_nonreencoding=use_hard_tokens,
        return_type=ReturnType.NEW_TEXT,
        model_kwargs=dict(),
    )
assert isinstance(adv_generator, AdversarialTextGenerationPipeline)

if use_hard_tokens:
    optim = GreedyCoordinateGradient(
        adv_generator.attack.parameters(),
        ignored_values=adv_generator.tokenizer.bad_token_ids,
        negative_only=False,
        n_tokens=1,
        coord_randk=0,
        coord_topk=256,
        global_topk=0,
    )
else:
    optim = Adam(adv_generator.attack.parameters(), lr=0.005)

 
 

def closure(prompts, completions, return_outputs=False):
    outputs: MutableMapping = adv_generator(prompts, completion=completions)
    if return_outputs:
        return outputs
    else:
        return torch.mean(torch.stack([output["loss"] for output in outputs]))


for prompts, completions in prompt_completion_batches: 
    for _ in (pbar := trange(max_steps)):
        with torch.inference_mode():

            adv_outputs = closure(prompts=prompts, completions=completions, return_outputs=True)
            adv_prompts = [adv_output["prompt_text"] for adv_output in adv_outputs]
            adv_completions = [adv_output["generated_text"] for adv_output in adv_outputs]
            loss = torch.mean(torch.stack([adv_output["loss"] for adv_output in adv_outputs]))

            loss.backward()
            with torch.inference_mode():
                if use_hard_tokens:
                    optim.step(closure)  # type: ignore
                else:
                    optim.step()  # type: ignore 

            pbar.set_postfix(loss=f"{loss:0.4f}")
 
    # extract prepended tokens
    with torch.inference_mode():
        model_inputs = adv_generator.preprocess("", completion="")  
        model_inputs = adv_generator.ensure_tensor_on_device(**model_inputs)
        adv_model_inputs = adv_generator.attack(model_inputs)
        # there is some question as to the right thing to do here
        if use_hard_tokens:
            # here adv_model_inputs should have what you need?
            print(f"Brandon inspection: adv_model_inputs keys are: {adv_model_inputs.keys()}")
        else:
            output_ids = generator.model.generate(  
                inputs_embeds=adv_model_inputs["inputs_embeds"],
                max_length=100,
                do_sample=False,
            )[0]
            decoded = generator.tokenizer.decode(output_ids)  # type: ignore

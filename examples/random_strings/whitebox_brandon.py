#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import numpy as np

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


def main(
    sequence: str,
    max_steps: int,
    num_tokens: int,
    lr: float,
    use_hard_tokens: bool,
):
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

    print(
        f"\nTrying to AVOID generating something starting with '{sequence}' of len={len(sequence)} | with: num_tokens({num_tokens})"
    )
    final_found, (adv_prefix, gen_completion) = train_defense(
        sequence=sequence,
        generator=generator,
        num_tokens=num_tokens,
        max_steps=max_steps,
        lr=lr,
        use_hard_tokens=use_hard_tokens,
    )
    


def train_defense(
    sequence: str,
    *,
    data_queries: list,
    data_responses: list,
    generator: Pipeline,
    num_tokens: int,
    max_steps: int,
    lr: float,
    use_hard_tokens: bool,
    seed: int = 2024, 
    batch_size: int = 20,
) -> tuple[bool, tuple[str, str]]:
    """
    sequence: string to make sure is not at the beginning of the response (~ startswith)
    data: list of string queries
    
    """

    # input validation
    assert len(data_queries) == len(data_responses), "Queries and responses must have the same length"

    torch.manual_seed(seed)
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
        optim = Adam(adv_generator.attack.parameters(), lr=lr)

    """
    This block is being replaced now that we are using prompts and responses
    prompt = ""
    adv_prompt = prompt
    adv_completion = []""
    found = False
    """
    adv_completions = batch_size * [""]

    num_batches = int(np.floor(len(data_queries) / batch_size))

    def closure(prompt_batch, completion_batch):
        outputs: list[MutableMapping] = [adv_generator(prompt, completion=completion)[0] for prompt, completion in zip(prompt_batch, completion_batch)]
        total_loss = outputs[0]["loss"]
        for idx in range(1, len(outputs)):
            total_loss += outputs[idx]["loss"]
        mean_loss = total_loss / len(outputs)
        # here testing whether the adversarial inputs were the same
        print(f"output keys: {outputs[0].keys()}")
        # print(f"\n#####\ntokens used were: \n{[output['prompt_text'] for output in outputs]} \n####")
        # Looking to see whether the attack parameters are changing
        # print(f"\n#####\nshape and values of attack parameters: {[(thingy.shape, thingy) for thingy in adv_generator.attack.parameters()]} \n#####\n")

        return outputs, mean_loss

    all_num_found = {}

    for step_num in (pbar := trange(max_steps)):
        """
        For each step we itate over all batches to improve tokens, then run with these fixed tokens 
        to get a train set evaluation by iterating again over all batches.
        """
        # First the epoch to improve tokens
        # TODO: Should I really remove partial batches like I am? Currently it is to have the same length of responses. Maybe just insure no partial batch
        print(f"\n####\nSTARTING step number: {step_num}\n####\n")
        num_found_this_step = 0
            
        for batch_idx in range(num_batches):
            batch_queries = data_queries[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_responses = data_responses[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            with torch.inference_mode():
                num_found = np.sum([not adv_completion.startswith(sequence) for adv_completion in adv_completions])
                num_found_this_step += num_found

            # we are providing the idea that the query is anwswered with the sequence (that we are trying to prevent). Then we'll do a gradient decent on the negative loss.
            adv_outputs, mean_loss = closure(prompt_batch=batch_queries, completion_batch=batch_responses)

            # The soft tokens contained within each of the list entries below should be the same as they all came from the same instance of adv_generator within the closure
            adv_prompts = [adv_output["prompt_text"] for adv_output in adv_outputs]
            adv_completions = [adv_output["generated_text"] for adv_output in adv_outputs]

            mean_loss.backward()
            with torch.inference_mode():
                if use_hard_tokens:
                    raise NotImplementedError("Hard tokens not implemented with this closure (returns a tuple)")
                    optim.step(closure)  # type: ignore
                else:
                    optim.step()  # type: ignore

            pbar.set_postfix(loss=f"{mean_loss:0.4f}")


            # Now try to inspect the soft tokens:
            model_inputs = adv_generator.preprocess("", completion="")  # type: ignore
            model_inputs = adv_generator.ensure_tensor_on_device(**model_inputs)
            adv_model_inputs = adv_generator.attack(model_inputs)  # type: ignore
            # print(f"\n# # # # \nShape and soft TOKENS: {adv_model_inputs['inputs_embeds'].shape, adv_model_inputs['inputs_embeds']}\n\n")

        all_num_found[step_num] = num_found_this_step
        print(f"\n\n\n####\nEND OF step number: {step_num} | num_found_this_step: {num_found_this_step}\n####\n\n\n")

    

    # Compute adversarial soft token embeddings
    model_inputs = adv_generator.preprocess("", completion="")  # type: ignore
    model_inputs = adv_generator.ensure_tensor_on_device(**model_inputs)
    adv_model_inputs = adv_generator.attack(model_inputs)  # type: ignore

    # Pass text or soft token embeddings to generator
    with torch.inference_mode():
        if use_hard_tokens:
            # This make sense because adv_prompts[idx] should be independent of idx
            # TODO: I have not tested this becuase I am not currently using hard tokens
            raise ValueError("You first need to insert a test.")
            output: MutableMapping = generator(adv_prompts[0])  # type: ignore
            decoded = output["generated_text"]
        else:
            soft_tokens_to_insert = adv_model_inputs["inputs_embeds"]
            output_ids = generator.model.generate(  # type: ignore[reportCallIssue]
                inputs_embeds=soft_tokens_to_insert,
                max_length=100,
                do_sample=False,
            )[0]
            decoded = generator.tokenizer.decode(output_ids)  # type: ignore

    return all_num_found, (adv_prompts[0], decoded), soft_tokens_to_insert  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequence",
        type=str,
        help="Target generation/completion.",
    )
    parser.add_argument(
        "--max_steps",
        dest="max_steps",
        type=int,
        default=5000,
        help="Optimise prompt for no more than N steps.",
    )
    parser.add_argument(
        "--num_tokens",
        dest="num_tokens",
        type=int,
        default=1,
        help="Number of optimised tokens",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.005,
        help="Learning rate for adversarial optimisation.",
    )
    parser.add_argument(
        "--use_hard_tokens",
        dest="use_hard_tokens",
        action="store_true",
        help="Find hard tokens instead of soft tokens in the emebdding space.",
    )

    args = parser.parse_args()
    if args.use_hard_tokens:
        print("WARN! Optimising hard tokens; lr will have no effect")

    main(
        args.sequence,
        args.max_steps,
        args.num_tokens,
        args.lr,
        args.use_hard_tokens,
    )

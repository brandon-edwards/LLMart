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
        f"\nTrying to generate '{sequence}' of len={len(sequence)} | with: num_tokens({num_tokens})"
    )
    found, (adv_prefix, gen_completion) = train_defense(
        sequence=sequence,
        generator=generator,
        num_tokens=num_tokens,
        max_steps=max_steps,
        lr=lr,
        use_hard_tokens=use_hard_tokens,
    )
    print(f"Final prompt: {repr(adv_prefix)} => {repr(gen_completion)}")
    if found:
        print("Found effective prompt")
    else:
        print("Failed to find effective prompt")


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
    adv_completion = ""
    found = False
    """

    def closure(prompt, completion, return_outputs=False):
        outputs: MutableMapping = adv_generator(prompt, completion=completion)[0]  # type: ignore
        return outputs if return_outputs else outputs["loss"]

    for step_num in (pbar := trange(max_steps)):
        """
        For each step we itate over all samples to improve tokens with the objective of answering ground truth.
        """
        print(f"\n####\nSTARTING step number: {step_num}\n####\n")
        with torch.inference_mode():
            
            for query, response in zip(data_queries, data_responses):
                
                adv_outputs: MutableMapping = closure(prompt=query, completion=response, return_outputs=True)
                adv_prompt = adv_outputs["prompt_text"]
                # don't need since we are not inspecting model outputs to see if we found good tokens adv_completion = adv_outputs["generated_text"]
                loss = adv_outputs["loss"]

                loss.backward()
                with torch.inference_mode():
                    if use_hard_tokens:
                        optim.step(closure)  # type: ignore
                    else:
                        optim.step()  # type: ignore

                pbar.set_postfix(loss=f"{loss:0.4f}")

                print(f"\n\n\n####\nEND OF step number: {step_num} | loss: {loss}\n####\n\n\n")

    

    # Compute adversarial soft token embeddings
    # TODO: Just trying something here, not sure if this is what we want (ask Sebastian)
    model_inputs = adv_generator.preprocess("", completion="")  # type: ignore
    model_inputs = adv_generator.ensure_tensor_on_device(**model_inputs)
    adv_model_inputs = adv_generator.attack(model_inputs)  # type: ignore

    # Pass text or soft token embeddings to generator
    with torch.inference_mode():
        if use_hard_tokens:
            output: MutableMapping = generator(adv_prompt)[0]  # type: ignore
            decoded = output["generated_text"]
        else:
            output_ids = generator.model.generate(  # type: ignore[reportCallIssue]
                inputs_embeds=adv_model_inputs["inputs_embeds"],
                max_length=100,
                do_sample=False,
            )[0]
            decoded = generator.tokenizer.decode(output_ids)  # type: ignore

    return decoded  # type: ignore


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

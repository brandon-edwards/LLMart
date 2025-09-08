#
# Copyright (C) 2025 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

import argparse
import numpy as np
import sys
import pickle as pkl

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


sys.path.append('/home/edwardsb/repositories/LLMart/examples/random_strings')
# This is now done outside of this notebook so that I can run it and walk away -- from whitebox_attack_data import attack as find_prepend_tokens_to_data
from brandon_utils import form_queries, form_responses, get_soft_token_defense_pickle_path, transfer_data_short_path
from brandon_utils import generate_nonrandom, pickled_manual_adv_data_path_bulk_train, get_generator, model_on_tokens, adv_success


def main(
     max_steps: int,
    num_tokens: int,
    lr: float,
    batch_size: int = 1,
    loss_sign: float = 1.0,
    use_hard_tokens: bool = False,
    device: str = "cpu",
    seed: int = 2024, 
    adv_data_path: str = pickled_manual_adv_data_path_bulk_train
):
    generator = get_generator(device=device)
    assert isinstance(generator.tokenizer, PreTrainedTokenizerBase)
    generator.tokenizer.pad_token = generator.tokenizer.eos_token

    ######## Get the data #######

    # grab the adversarial data from the pickle file (NOTE: As long as the default: pickled_manual_adv_data_path_bulk_train is used above,
    # this is constructed in the notebook: manual_adv_attack_and_save.ipynb). The format of the pickled object is different here than the last adv data was since adversarial
    # completions and adversarial priompts are not as relevant here (since the objective is to get the model to output an empty string and only the ones that succeeded are kept).
    with open(adv_data_path, 'rb') as _f:
        (indices, adversarial_data) = pkl.load(_f)

    # We are going to also need the transfer learning data
    with open(transfer_data_short_path, 'rb') as _f:
        transfer_data_short = pkl.load(_f)
    transfer_data_matching_adv_samples = [sample for idx, sample in enumerate(transfer_data_short) if idx in indices]

    # validate using instruction field (so partial validation)
    for adv_dict, trans_dict in zip(adversarial_data, transfer_data_matching_adv_samples):
        assert adv_dict['instruction'] == trans_dict['instruction'], f"Missmatch in instruction field, {adv_dict['instruction']} != {trans_dict['instruction']}"

    # manually restricting to 200 samples here
    print(f"BRANDON TODO: increase to greater than 200 samples !!!!!!!!!!!!!!!")

    # TODO: fix this to be larger
    adversarial_data = adversarial_data[:200]

    assert len(adversarial_data) == len(transfer_data_matching_adv_samples), f"Adversarial data and transfer learning data must be the same length but have length {len(adversarial_data)} and {len(transfer_data_matching_adv_samples)} respectively"

    ######## Form the queries and responses ######
    adv_data_queries = form_queries(adversarial_data) 
    data_tl_responses = form_responses(transfer_data_matching_adv_samples) # transfer learning responses

    soft_token_defense_pickle_path = get_soft_token_defense_pickle_path(num_tokens=num_tokens,
                                                                        max_steps=max_steps,
                                                                        lr=lr,
                                                                        seed=seed, 
                                                                        batch_size=batch_size,
                                                                        loss_sign=loss_sign,
                                                                        use_hard_tokens=use_hard_tokens
                                                                    )

    (adv_prompt, decoded), soft_tokens_to_insert, mean_loss = train_defense(
    data_queries=adv_data_queries,
    data_responses=data_tl_responses,
    max_steps=max_steps,
    num_tokens=num_tokens,
    lr=lr,
    use_hard_tokens=use_hard_tokens,
    generator=generator,
    batch_size=batch_size,
    seed=seed
)
    # Now save results to the soft token defense pickle file
    with open(soft_token_defense_pickle_path, "wb") as _file:
        pkl.dump(((adv_prompt, decoded), soft_tokens_to_insert, mean_loss), _file)


def train_defense(
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
    loss_sign: float = 1.0,
) -> tuple[bool, tuple[str, str]]:
    """
    data_queries: list of string queries
    data_responses: list of string desired responses


    loss_sign: -1.0 to disincentivise the model to generate the sequence at the start of the response, 1.0 to incentivise it to generate the sequence at the start of the response.
    
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

    num_batches = int(np.floor(len(data_queries) / batch_size))

    def closure(prompt_batch, completion_batch, loss_sign: float = 1.0):
        outputs: list[MutableMapping] = [adv_generator(prompt, completion=completion)[0] for prompt, completion in zip(prompt_batch, completion_batch)]
        total_loss = loss_sign * outputs[0]["loss"]
        for idx in range(1, len(outputs)):
            total_loss += loss_sign * outputs[idx]["loss"]
        mean_loss = total_loss / len(outputs)
        # here testing whether the adversarial inputs were the same
        # print(f"output keys: {outputs[0].keys()}")
        # print(f"\n#####\ntokens used were: \n{[output['prompt_text'] for output in outputs]} \n####")
        # Looking to see whether the attack parameters are changing
        # print(f"\n#####\nshape and values of attack parameters: {[(thingy.shape, thingy) for thingy in adv_generator.attack.parameters()]} \n#####\n")

        return outputs, mean_loss

    for step_num in (pbar := trange(max_steps)):
        """
        For each step we itate over all batches to improve tokens, then run with these fixed tokens 
        to get a train set evaluation by iterating again over all batches.
        """
            
        for batch_idx in range(num_batches):
            batch_queries = data_queries[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_responses = data_responses[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # print(f"\nBrandon DEBUG - about to compute closure on:")
            # print(f"Brandon DEBUG - batch_queries: {batch_queries}")
            print(f"Brandon DEBUG - batch_responses: {batch_responses}\n")
             # we are providing the idea that the query is anwswered with the sequence (that we are trying to prevent). Then we'll do a gradient decent on the negative loss.
            adv_outputs, mean_loss = closure(prompt_batch=batch_queries, completion_batch=batch_responses, loss_sign=loss_sign)

            # print(f"Brandon DEBUG - adv_outpus has keys: {adv_outputs[0].keys()}")   # these are the keys: ['generated_text', 'prompt_text', 'loss', 'input_ids', 'logits', 'labels']
            print(f"Brandon DEBUG - generated text for batch_idx:{batch_idx} is: {adv_outputs[0]['generated_text']}")
            # print(f"Brandon DEBUG - prompt text for batch_idx:{batch_idx} is: {adv_outputs[0]['prompt_text']}")
            # print(f"Brandon DEBUG - labels for batch_idx:{batch_idx} is: {adv_outputs[0]['labels']}")
            # print(f"Brandon DEBUG - input_ids for batch_idx:{batch_idx} is: {adv_outputs[0]['input_ids']}")
            # print(f"Brandon DEBUG - mean_loss for batch_idx:{batch_idx} is: {mean_loss}")
            # print(f"Brandon DEBUG - length of adv_outputs is: {len(adv_outputs)}") # was consistently 1 when tested 8/22/2025

            # The soft tokens contained within each of the list entries below should be the same as they all came from the same instance of adv_generator within the closure
            adv_prompts = [adv_output["prompt_text"] for adv_output in adv_outputs]

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

    return (adv_prompts[0], decoded), soft_tokens_to_insert, mean_loss  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        default=10,
        help="Number of optimised tokens",
    )
    parser.add_argument(
        "--lr",
        dest="lr",
        type=float,
        default=0.0005,
        help="Learning rate for adversarial optimisation.",
    )
    parser.add_argument(
        "--use_hard_tokens",
        dest="use_hard_tokens",
        action="store_true",
        help="Find hard tokens instead of soft tokens in the emebdding space.",
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        default=1,
        help="Batch size for adversarial optimisation.",
    )
    parser.add_argument(
        "--loss_sign",
        dest="loss_sign",
        type=float,
        default=1.0,
        help="Loss sign for adversarial optimisation.",
    )
    parser.add_argument(
        "--device",
        dest="device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g. 'cpu' or 'cuda:0' - note device num will be designated by launch bash script).",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=2024,
        help="Random seed for initialization.",
    )

    args = parser.parse_args()
    if args.use_hard_tokens:
        print("WARN! Optimising hard tokens; lr will have no effect")

    main(
        max_steps=args.max_steps,
        num_tokens=args.num_tokens,
        lr=args.lr,
        batch_size=args.batch_size,
        loss_sign=args.loss_sign,
        use_hard_tokens=args.use_hard_tokens,
        device=args.device,
        seed=args.seed
    )

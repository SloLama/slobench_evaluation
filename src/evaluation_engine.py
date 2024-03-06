import json
import os
from argparse import ArgumentParser
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import logging

from prompt_creation import *
from evaluators import *

MODELS_DIR = "/ceph/hpc/data/st2311-ponj-users/models"
HT_DATA_DIR = "/ceph/hpc/data/st2311-ponj-users/slobench/SuperGLUE-HumanT/csv"
MT_DATA_DIR = "/ceph/hpc/data/st2311-ponj-users/slobench/SuperGLUE-GoogleMT/csv"
SUPPORTED_DATASETS = [
    "BoolQ"
]


def load_model(model_path, f_out) -> MegatronGPTModel:
    trainer = Trainer(strategy=NLPDDPStrategy(), accelerator="gpu", devices=1)
    save_restore_connector = NLPSaveRestoreConnector()

    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, model_path)

    model = MegatronGPTModel.restore_from(
        restore_path=model_path,
        trainer=trainer,
        save_restore_connector=save_restore_connector,
        map_location=f'cuda:{trainer.local_rank}'
    )

    model.freeze()

    f_out.write(f"Model: {model_path}\n")
    f_out.write(f"Model config: {model.cfg}\n\n")

    return model


def load_data(dataset, load_ht, load_mt):
    assert (
            load_ht or load_mt
    ), "Loading MT and HT are both set to False. At least one must be set to True."

    if not load_ht:
        warnings.warn(
            f"Loading human translated data is set to False for {dataset}. Loading only machine translated data.")
    if not load_mt:
        warnings.warn(
            f"Loading machine translated data is set to False for {dataset}. Loading only human translated data.")

    train_data, eval_data = None, None

    logging.info(f"Loading {dataset} data.")

    if load_ht:
        train_data = pd.read_csv(os.path.join(HT_DATA_DIR, dataset, "train.csv"), index_col="idx")
        eval_data = pd.read_csv(os.path.join(HT_DATA_DIR, dataset, "val.csv"), index_col="idx")

        logging.info(f"Number of human translated train examples: {train_data.shape[0]}")
        logging.info(f"Number of human translated evaluation examples: {eval_data.shape[0]}")

    if load_mt:
        train_data_mt = pd.read_csv(os.path.join(MT_DATA_DIR, dataset, "train.csv"), index_col="idx")
        eval_data_mt = pd.read_csv(os.path.join(MT_DATA_DIR, dataset, "val.csv"), index_col="idx")

        logging.info(f"Number of machine translated train examples: {train_data_mt.shape[0]}")
        logging.info(f"Number of machine translated evaluation examples: {eval_data_mt.shape[0]}")

        if train_data is not None:
            # Replace machine translated rows with human translated ones
            train_data_mt = train_data_mt.drop(train_data.index)
            eval_data_mt = eval_data_mt.drop(eval_data.index)

            train_data = pd.concat([train_data, train_data_mt], axis=0)
            eval_data = pd.concat([eval_data, eval_data_mt], axis=0)

        else:
            train_data = train_data_mt
            eval_data = eval_data_mt

    logging.info(f"Number of train examples: {train_data.shape[0]}")
    logging.info(f"Number of evaluation examples: {eval_data.shape[0]}")

    return eval_data, train_data


def get_prompt_creator(dataset, train_data, seed) -> SloBenchPromptCreator:
    if dataset == "BoolQ":
        return BoolQPromptCreator(seed, train_data)


def get_evaluator(dataset, f_out) -> SloBenchEvaluator:
    if dataset == "BoolQ":
        return BoolQEvaluator(f_out)


def get_sampling_and_length_params(dataset):
    if dataset == "BoolQ":
        sampling_params = {
            "use_greedy": False,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "add_BOS": False,
            "all_probs": False,
            "compute_logprob": False,
            "compute_attention_mask": False,
            "end_strings": ['</s>']
        }

        length_params = {"min_length": 0, "max_length": 2}

    return sampling_params, length_params


def run_engine(config, output_file):
    f_out = open(output_file, "w")

    model = load_model(config["model"], f_out)
    benchmarks = config["benchmarks"]

    for benchmark in benchmarks:
        dataset = benchmark["dataset"]
        assert (
                dataset in SUPPORTED_DATASETS
        ), f'{dataset} is not supported. Currently supported datasets: {SUPPORTED_DATASETS}'

        f_out.write(f"---------------------- {dataset} evaluation ----------------------\n")

        load_ht = benchmark.get("human_translated", False)
        load_mt = benchmark.get("machine_translated", False)
        eval_data, train_data = load_data(dataset, load_ht, load_mt)

        seed = benchmark.get("seed", 42)
        prompt_creator = get_prompt_creator(dataset, train_data, seed)

        evaluator = get_evaluator(dataset, f_out)

        true_labels = prompt_creator.get_labels(eval_data)

        evaluator.compute_general_stats(true_labels)

        k_list = benchmark["k"]
        evaluation_params = benchmark["evaluation"]

        sampling_params, length_params = get_sampling_and_length_params(dataset)

        for k in k_list:
            if k != 0:
                majority_labels = []
                last_labels = []
            predictions = []

            for idx in tqdm(eval_data.index):
                example = eval_data.loc[idx]
                if k == 0:
                    prompt = prompt_creator.create_zero_shot_prompt(example)
                else:
                    prompt, example_labels = prompt_creator.create_few_shot_prompt(example, k)
                    majority_labels.append(prompt_creator.get_majority_label(example_labels))
                    last_labels.append(example_labels[-1])

                try:
                    prediction = model.generate(
                        [prompt],
                        length_params=length_params,
                        sampling_params=sampling_params
                    )["sentences"][0]
                except:
                    warnings.warn(f"An error occured while generating response for the following prompt: {prompt}")
                    prediction = "An error ocured during generation. Invalid prediction."

                predictions.append(prediction)

            logging.info(f"Running {k}-shot evaluation for {dataset}")
            f_out.write(f"\nResults for {k}-shot experiment:\n")
            if k == 0:
                evaluator.evaluate(evaluation_params, predictions, true_labels)
            else:
                majority_labels = np.array(majority_labels)
                if k == 1:
                    last_labels = None
                else:
                    last_labels = np.array(last_labels)
                evaluator.evaluate(evaluation_params, predictions, true_labels, majority_labels, last_labels)

        f_out.write("------------------------------------------------------\n\n")

    f_out.close()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str, required=True,
                           help="Path to the JSON file containing evaluation configuration.")
    argparser.add_argument("--output_file", type=str, required=True, help="Path to the output file.")
    args = argparser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    run_engine(config, args.output_file)

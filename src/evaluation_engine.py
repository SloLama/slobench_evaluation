import json
from argparse import ArgumentParser
import warnings

from tqdm import tqdm

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from pytorch_lightning.trainer.trainer import Trainer
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.utils import logging

from evaluators import *
from data_loaders import *

MODELS_DIR = "/ceph/hpc/data/st2311-ponj-users/models"
SUPPORTED_DATASETS = [
    "BoolQ",
    "MultiRC",
    "WSC",
    "WSC_generative",
    "COPA",
    "RTE",
    "CB",
    "NLI"
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


def load_data(dataset, load_ht, load_mt, seed, prompt_template, prefix) -> SloBenchDataLoader:
    assert (
            dataset=="NLI" or load_ht or load_mt
    ), "Loading MT and HT are both set to False. At least one must be set to True."

    if not (dataset == "NLI" or load_ht):
        warnings.warn(
            f"Loading human translated data is set to False for {dataset}. Loading only machine translated data.")
    if not (dataset in ["WSC", "WSC_generative", "NLI"] or load_mt):
        warnings.warn(
            f"Loading machine translated data is set to False for {dataset}. Loading only human translated data.")

    if dataset in ["WSC", "WSC_generative"]:
        warnings.warn(
            "Ignoring config values for machine translated and human translated data as WSC includes only human translated data.")

    if dataset == "BoolQ":
        data_loader = BoolQDataLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "MultiRC":
        data_loader = MultiRCDataLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "WSC":
        data_loader = WSCDataLoader(human_translated=True, machine_translated=False, seed=seed, prompt_template=prompt_template, prefix=prefix)
    elif dataset == "WSC_generative":
        data_loader = WSCGenerativeDataLoader(human_translated=True, machine_translated=False, seed=seed, prompt_template=prompt_template, prefix=prefix)
    elif dataset == "COPA":
        data_loader = COPADataLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "RTE":
        data_loader = RTEDataLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "CB":
        data_loader = CBDataLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "NLI":
        data_loader = NLILoader(None, None, seed, prompt_template, prefix)

    logging.info(f"Loading {dataset} data.")
    data_loader.load_data()

    logging.info(f"Number of train examples: {data_loader.train_data_size()}")
    logging.info(f"Number of evaluation examples: {data_loader.eval_data_size()}")

    return data_loader


def get_evaluator(dataset, f_out) -> SloBenchEvaluator:
    if dataset == "BoolQ":
        return BoolQEvaluator(f_out)
    if dataset == "MultiRC":
        return MultiRCEvaluator(f_out)
    if dataset == "WSC":
        return WSCEvaluator(f_out)
    if dataset == "WSC_generative":
        return WSCGenerativeEvaluator(f_out)
    if dataset == "COPA":
        return COPAEvaluator(f_out)
    if dataset == "RTE":
        return RTEEvaluator(f_out)
    if dataset == "CB":
        return CBEvaluator(f_out)
    if dataset == "NLI":
        return NLIEvaluator(f_out)


def get_sampling_and_length_params(dataset):
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

    if dataset in ["BoolQ", "WSC", "COPA"]:
        length_params = {"min_length": 0, "max_length": 5}
    elif dataset == "MultiRC":
        length_params = {"min_length": 0, "max_length": 50}
    elif dataset == "WSC_generative":
        length_params = {"min_length": 0, "max_length": 20}
    elif dataset in ["RTE", "CB", "NLI"]:
        length_params = {"min_length": 0, "max_length": 10}

    # Add some end strings
    if dataset in ["BoolQ", "WSC"]:
        sampling_params["end_strings"] += ["Da", "da", "Ne", "ne", "Da.", "da.", "Ne.", "ne.", "Da,", "da,", "Ne,", "ne,"]
    elif dataset == "COPA":
        sampling_params["end_strings"] += ["1", "2"]
    elif dataset == "RTE":
        sampling_params["end_strings"] += ["Drži", "drži", "Ne drži", "ne drži", "Drži.", "drži.", "Ne drži.", "ne drži."]
    elif dataset == "CB":
        sampling_params["end_strings"] += ["Drži", "drži", "Ne drži", "ne drži", "Ne vemo", "ne vemo", "Drži.", "drži.", "Ne drži.", "ne drži.", "Ne vemo.", "ne vemo."]
    elif dataset == "NLI":
        sampling_params["end_strings"] += ["Sosledje", "sosledje", "Nasprotovanje", "nasprotovanje", "Nevtralnost", "nevtralnost", "Sosledje.", "sosledje.", "Nasprotovanje.", "nasprotovanje.", "Nevtralnost.", "nevtralnost."]

    return sampling_params, length_params


def run_engine(config, output_file):
    f_out = open(output_file, "w")

    model = load_model(config["model"], f_out)
    benchmarks = config["benchmarks"]

    default_template = "{instruction}\n\n{input}\n"
    prompt_template = config.get("prompt_template", default_template)

    for benchmark in benchmarks:
        dataset = benchmark["dataset"]
        assert (
                dataset in SUPPORTED_DATASETS
        ), f'{dataset} is not supported. Currently supported datasets: {SUPPORTED_DATASETS}'

        f_out.write(f"---------------------- {dataset} evaluation ----------------------\n")

        load_ht = benchmark.get("human_translated", False)
        load_mt = benchmark.get("machine_translated", False)
        seed = benchmark.get("seed", 42)
        prefix = benchmark.get("prefix", None)
        data_loader = load_data(dataset, load_ht, load_mt, seed, prompt_template, prefix)

        evaluator = get_evaluator(dataset, f_out)

        true_labels = data_loader.get_eval_labels()

        evaluator.compute_general_stats(true_labels)

        k_list = benchmark["k"]
        evaluation_params = benchmark["evaluation"]

        sampling_params, length_params = get_sampling_and_length_params(dataset)

        for k in k_list:
            if k != 0:
                majority_labels = []
                last_labels = []
            predictions = []

            for prompt, majority_label, last_label in tqdm(data_loader.get_eval_data_iterator(k), total=data_loader.eval_data_size()):
                if k > 0:
                    majority_labels.append(majority_label)
                    last_labels.append(last_label)

                try:
                    prediction = model.generate(
                        [prompt],
                        length_params=length_params,
                        sampling_params=sampling_params
                    )["sentences"][0]

                    # Remove prompt from prediction
                    prediction = prediction[len(prompt):]
                except:
                    warnings.warn(f"An error occured while generating response for the following prompt: {prompt}")
                    prediction = "An error ocured during generation. Invalid prediction."

                predictions.append(prediction)

            logging.info(f"Running {k}-shot evaluation for {dataset}")
            f_out.write(f"\nResults for {k}-shot experiment:\n")
            if k == 0:
                evaluator.evaluate(evaluation_params, predictions, true_labels)
            else:
                try:
                    majority_labels = np.array(majority_labels)
                except:
                    majority_labels = np.array(majority_labels, dtype=object)
                if k == 1 and dataset != "WSC_generative":
                    last_labels = None
                else:
                    try:
                        last_labels = np.array(last_labels)
                    except:
                        last_labels = np.array(last_labels, dtype=object)
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

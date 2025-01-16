import json
from argparse import ArgumentParser
import warnings
from math import ceil

from tqdm import tqdm
from torch.utils.data import DataLoader as Torch_DL

from evaluators import *
from data_loaders import *
from model_wrappers import *

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


def load_data(dataset, load_ht, load_mt, seed, prompt_template, instruction, prefix) -> SloBenchDataLoader:
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
        data_loader = BoolQDataLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "MultiRC":
        data_loader = MultiRCDataLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "WSC":
        data_loader = WSCDataLoader(human_translated=True, machine_translated=False, seed=seed, prompt_template=prompt_template, instruction=instruction, prefix=prefix)
    elif dataset == "WSC_generative":
        data_loader = WSCGenerativeDataLoader(human_translated=True, machine_translated=False, seed=seed, prompt_template=prompt_template, instruction=instruction, prefix=prefix)
    elif dataset == "COPA":
        data_loader = COPADataLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "RTE":
        data_loader = RTEDataLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "CB":
        data_loader = CBDataLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "NLI":
        data_loader = NLILoader(None, None, seed, prompt_template, instruction, prefix)

    print(f"Loading {dataset} data.")
    data_loader.load_data()

    print(f"Number of train examples: {data_loader.train_data_size()}")
    print(f"Number of evaluation examples: {data_loader.eval_data_size()}")

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


def run_engine(config, output_file):
    f_out = open(output_file, "w")

    # get batch size
    batch_size = config["batch_size"]

    # load model
    model_library = config["model"]["library"]
    if model_library == "nemo":
        model = NemoModelWrapper(config["model"]["path"])
    elif model_library == "huggingface":
        model = HFModelWrapper(config["model"]["path"], config["model"].get("apply_chat_template", True), batch_size)
    elif model_library.lower() == "vllm":
        model = VLLMModelWrapper(config["model"]["path"], config["model"].get("apply_chat_template", True))
    else:
        raise ValueError('Unsupported model library. Only supported libraries are "nemo", "huggingface", and "vllm"')
    model.print_model_info(f_out)
    benchmarks = config["benchmarks"]

    # get prompt schemes
    with open(config["prompt_scheme_file"], "r", encoding="utf-8") as scheme_file:
        prompt_schemes = json.load(scheme_file)

    # go through all included benchmarks
    for benchmark in benchmarks:
        dataset = benchmark["dataset"]
        assert (
                dataset in SUPPORTED_DATASETS
        ), f'{dataset} is not supported. Currently supported datasets: {SUPPORTED_DATASETS}'

        load_ht = benchmark.get("human_translated", False)
        load_mt = benchmark.get("machine_translated", False)
        seed = benchmark.get("seed", 42)

        f_out.write(f"---------------------- {dataset} evaluation ----------------------\n\n")

        # go through all included prompt schemes
        for scheme_id, prompt_scheme in enumerate(prompt_schemes[dataset]):
            f_out.write(f"============== prompt scheme {scheme_id} ==============\n")

            default_template = "{instruction}\n\n{input}\n"
            prompt_template = prompt_scheme.get("prompt_template", default_template)
            instruction = prompt_scheme["instruction"]
            prefix = prompt_scheme.get("prefix", None)

            data_loader = load_data(dataset, load_ht, load_mt, seed, prompt_template, instruction, prefix)

            evaluator = get_evaluator(dataset, f_out)

            true_labels = data_loader.get_eval_labels()

            evaluator.compute_general_stats(true_labels)

            k_list = prompt_scheme["k"]
            evaluation_params = benchmark["evaluation"]

            model.set_generation_params(dataset)

            for k in k_list:
                if k != 0:
                    majority_labels = []
                    last_labels = []
                predictions = []

                # build list of prompts
                prompts = []
                for prompt, majority_label, last_label in data_loader.get_eval_data_iterator(k):
                    if k > 0:
                        majority_labels.append(majority_label)
                        last_labels.append(last_label)

                    prompts.append(prompt)

                # Split prompts into batches
                batches = Torch_DL(prompts, batch_size=batch_size)

                # run prediction for every batch
                for batch in tqdm(batches, total=ceil(data_loader.eval_data_size()/batch_size)):
                    try:
                        batch_prediction = model.generate(batch)

                    except Exception as ex:
                        warnings.warn(f"An error occured while generating responses for one of the batches: {ex}")
                        batch_prediction = ["An error ocured during generation. Invalid prediction."] * len(batch)

                    predictions.extend(batch_prediction)

                print(f"Running {k}-shot evaluation for {dataset}")
                f_out.write(f"\nFull prompt scheme used:\n")
                for key, value in prompt_scheme.items():
                    f_out.write(f"{str(key)} : {str(value)}\n")
                f_out.write("\n")
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

            f_out.write("=========================================\n\n")

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

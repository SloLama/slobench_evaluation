import json
from argparse import ArgumentParser
import warnings
from math import ceil

from tqdm import tqdm
from torch.utils.data import DataLoader as Torch_DL

from submission_creators import *
from data_loaders import *
from model_wrappers import *

SUPPORTED_DATASETS = [
    "BoolQ",
    "MultiRC",
    "WSC",
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
    if not (dataset in ["WSC", "NLI"] or load_mt):
        warnings.warn(
            f"Loading machine translated data is set to False for {dataset}. Loading only human translated data.")

    if dataset in ["WSC"]:
        warnings.warn(
            "Ignoring config values for machine translated and human translated data as WSC includes only human translated data.")

    if dataset == "BoolQ":
        data_loader = BoolQTestLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "MultiRC":
        data_loader = MultiRCTestLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "WSC":
        data_loader = WSCTestLoader(human_translated=True, machine_translated=False, seed=seed, prompt_template=prompt_template, instruction=instruction, prefix=prefix)
    elif dataset == "COPA":
        data_loader = COPATestLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "RTE":
        data_loader = RTETestLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "CB":
        data_loader = CBTestLoader(load_ht, load_mt, seed, prompt_template, instruction, prefix)
    elif dataset == "NLI":
        data_loader = NLITestDataLoader(None, None, seed, prompt_template, instruction, prefix)

    print(f"Loading {dataset} data.")
    data_loader.load_data()

    print(f"Number of train examples: {data_loader.train_data_size()}")
    print(f"Number of test examples: {data_loader.eval_data_size()}")

    return data_loader


def get_creator(dataset, output_dir) -> SlobenchSubmissionCreator:
    if dataset == "BoolQ":
        return BoolQSubmissionCreator(output_dir)
    if dataset == "CB":
        return CBSubmissionCreator(output_dir)
    if dataset == "COPA":
        return COPASubmissionCreator(output_dir)
    if dataset == "MultiRC":
        return MultiRCSubmissionCreator(output_dir)
    if dataset == "RTE":
        return RTESubmissionCreator(output_dir)
    if dataset == "WSC":
        return WSCSubmissionCreator(output_dir)
    if dataset == "NLI":
        return NLISubmissionCreator(output_dir)


def prepare_submission(config, output_dir):
    # get batch size
    batch_size = config["batch_size"]

    # load model
    model_library = config["model"]["library"]
    if model_library == "nemo":
        model = NemoModelWrapper(config["model"]["path"])
    elif model_library == "huggingface":
        model = HFModelWrapper(config["model"]["path"], config["model"].get("apply_chat_template", True), batch_size=batch_size)
    elif model_library.lower() == "vllm":
        model = VLLMModelWrapper(config["model"]["path"], config["model"].get("apply_chat_template", True))
    else:
        raise ValueError('Unsupported model library. Only supported libraries are "nemo", "huggingface", and "vllm"')
    benchmarks = config["benchmarks"]

    os.makedirs(output_dir, exist_ok=True)

    # get prompt schemes
    with open(config["prompt_scheme_file"], "r", encoding="utf-8") as scheme_file:
        prompt_schemes = json.load(scheme_file)

    for benchmark in benchmarks:
        dataset = benchmark["dataset"]
        assert (
                dataset in SUPPORTED_DATASETS
        ), f'{dataset} is not supported. Currently supported datasets: {SUPPORTED_DATASETS}'

        model.set_generation_params(dataset)

        load_ht = benchmark.get("human_translated", False)
        load_mt = benchmark.get("machine_translated", False)
        seed = benchmark.get("seed", 42)

        # load the prompt scheme. For submission preparation, only the first scheme for every benchmark in the file
        # will be used
        default_template = "{instruction}\n\n{input}\n"
        prompt_template = prompt_schemes[dataset][0].get("prompt_template", default_template)
        instruction = prompt_schemes[dataset][0]["instruction"]
        prefix = prompt_schemes[dataset][0].get("prefix", None)
        data_loader = load_data(dataset, load_ht, load_mt, seed, prompt_template, instruction, prefix)
        k = prompt_schemes[dataset][0]["k"][0]

        creator = get_creator(dataset, output_dir)
        predictions = []

        # build list of prompts
        prompts = []
        for prompt, _, _ in data_loader.get_eval_data_iterator(k):
            prompts.append(prompt)

        # Split data into batches
        batches = Torch_DL(prompts, batch_size=batch_size)

        print(f"Processing {dataset} predictions ...")
        for batch in tqdm(batches, total=ceil(data_loader.eval_data_size()/batch_size)):
            try:
                batch_prediction = model.generate(batch)

            except Exception as ex:
                warnings.warn(f"An error occured while generating responses for one of the batches: {ex}")

                batch_prediction = ["An error ocured during generation. Invalid prediction."] * batch_size

            predictions.extend(batch_prediction)

        data_info = [creator.get_data_info(instance) for instance in data_loader._eval_iter()]

        creator.prepare_submission(predictions, data_info)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config", type=str, required=True,
                           help="Path to the JSON file containing benchmark configuration.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the dir where output files will be saved.")
    args = argparser.parse_args()

    with open(args.config, "r") as file:
        config = json.load(file)

    prepare_submission(config, args.output_dir)

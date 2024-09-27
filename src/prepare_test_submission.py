import json
from argparse import ArgumentParser
import warnings

from tqdm import tqdm

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


def load_data(dataset, load_ht, load_mt, seed, prompt_template, prefix) -> SloBenchDataLoader:
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
        data_loader = BoolQTestLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "MultiRC":
        data_loader = MultiRCTestLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "WSC":
        data_loader = WSCTestLoader(human_translated=True, machine_translated=False, seed=seed, prompt_template=prompt_template, prefix=prefix)
    elif dataset == "COPA":
        data_loader = COPATestLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "RTE":
        data_loader = RTETestLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "CB":
        data_loader = CBTestLoader(load_ht, load_mt, seed, prompt_template, prefix)
    elif dataset == "NLI":
        data_loader = NLITestDataLoader(None, None, seed, prompt_template, prefix)

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
    model_library = config["model"]["library"]
    if model_library == "nemo":
        model = NemoModelWrapper(config["model"]["path"])
    elif model_library == "huggingface":
        model = HFModelWrapper(config["model"]["path"], config["model"].get("chat", True))
    else:
        raise ValueError('Unsupported model library. Only supported libraries are "nemo" and "huggingface"')
    benchmarks = config["benchmarks"]

    os.makedirs(output_dir, exist_ok=True)

    default_template = "{instruction}\n\n{input}\n"
    prompt_template = config.get("prompt_template", default_template)

    for benchmark in benchmarks:
        dataset = benchmark["dataset"]
        assert (
                dataset in SUPPORTED_DATASETS
        ), f'{dataset} is not supported. Currently supported datasets: {SUPPORTED_DATASETS}'

        model.set_generation_params(dataset)

        load_ht = benchmark.get("human_translated", False)
        load_mt = benchmark.get("machine_translated", False)
        seed = benchmark.get("seed", 42)
        prefix = benchmark.get("prefix", None)
        data_loader = load_data(dataset, load_ht, load_mt, seed, prompt_template, prefix)
        k = benchmark.get("k", 0)

        creator = get_creator(dataset, output_dir)
        predictions = []

        print(f"Processing {dataset} predictions ...")
        for prompt, _, _ in tqdm(data_loader.get_eval_data_iterator(k),
                                                       total=data_loader.eval_data_size()):
            try:
                prediction = model.generate(prompt)

            except:
                warnings.warn(f"An error occured while generating response for the following prompt: {prompt}")
                prediction = "An error ocured during generation. Invalid prediction."

            predictions.append(prediction)

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

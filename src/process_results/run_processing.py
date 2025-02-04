import os
from argparse import ArgumentParser
import pandas as pd

from process_results.dataset_processors import *


def get_dataset_metrics(dataset):
    if dataset in ["BoolQ", "WSC", "COPA", "RTE"]:
        return ["accuracy"]

    elif dataset == "MultiRC":
        return ["exact_match", "per_question_f1", "f1_over_all_answers"]

    elif dataset == "WSC_generative":
        return ["accuracy", "levenshtein_distance"]

    elif dataset == "CB":
        return ["accuracy", "f1"]

    elif dataset == "NLI":
        return ["accuracy", "precision_entailment", "recall_entailment", "f1_entailment", "precision_neutral", "recall_neutral", "f1_neutral", "precision_contradiction", "recall_contradiction", "f1_contradiction"]

    else:
        raise ValueError("Unsupported dataset.", dataset)


def get_dataset_processor(dataset: str) -> DatasetProcessor:
    if dataset == "BoolQ":
        return BoolQProcessor()
    
    elif dataset == "MultiRC":
        return MultiRCProcessor()
    
    elif dataset == "WSC":
        return WSCProcessor()
    
    elif dataset == "WSC_generative":
        return WSCGenerativeProcessor()
    
    elif dataset == "COPA":
        return COPAProcessor()
    
    elif dataset == "RTE":
        return RTEProcessor()
    
    elif dataset == "CB":
        return CBProcessor()
    
    elif dataset == "NLI":
        return NLIProcessor()

    else:
        raise ValueError("Unsupported dataset.", dataset)
    

def results_to_file(results, output_file, datasets):
    metrics_columns = []
    for dataset in datasets:
        for metric in get_dataset_metrics(dataset):
            metrics_columns.append(dataset + "_" + metric)
    invalid_predictions_columns = [dataset + "_" + "invalid_predictions" for dataset in datasets]
    correlation_columns = []
    for dataset in datasets:
        if dataset == "WSC_generative":
            continue
        for correlation in ["last_example_correlation", "majority_correlation"]:
            correlation_columns.append(dataset + "_" + correlation)

    for k, model_results in results.items():
        data = {"Model": []}
        for metric in metrics_columns:
            data[metric] = []
        for ip_col in invalid_predictions_columns:
            data[ip_col] = []
        for correlation in correlation_columns:
            data[correlation] = []

        for model, dataset_results in model_results.items():
            data["Model"].append(model)
            for dataset, result in dataset_results.items():
                for col, val in result.items():
                    data[dataset + "_" + col].append(val)

        data = pd.DataFrame(data)
        print(20*"-" + f" {k}-shot results " + 20*"-")

        print("Metrics:")
        metric_data = data[["Model"] + metrics_columns]
        print(metric_data.to_markdown())
        metric_data.to_csv(f"{output_file}_{k}_metrics.csv", index=False)
        print()

        print("Invalid predictions:")
        ip_data = data[["Model"] + invalid_predictions_columns]
        print(ip_data.to_markdown())
        ip_data.to_csv(f"{output_file}_{k}_invalid_predictions.csv", index=False)
        print()

        if k > 0:
            print("Correlation with few-shot examples:")
            correlation_data = data[["Model"] + correlation_columns]
            print(correlation_data.to_markdown())
            correlation_data.to_csv(f"{output_file}_{k}_correlations.csv", index=False)

        print(50 * "-")
        print()


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input file/dir containing the results. If the dir is provided all files inside are processed."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="BoolQ,MultiRC,WSC,WSC_generative,COPA,RTE,CB,NLI"
    )

    return parser.parse_args()
    

def run_processing(input_path, output_file, datasets):
    k_results = {}

    if os.path.isdir(input_path):
        input_files = [os.path.join(input_path, filename) for filename in os.listdir(input_path) if filename.endswith(".txt")]
    else:
        input_files = [input_path]

    for input_file in input_files:
        f_in = open(input_file, "r")
        model_name = f_in.readline()[len("Model:"):].strip()

        result_processor = None
        result_lines = None
        for line in f_in:
            line = line.strip()

            if line.startswith("---"):
                line = line.strip("- ")

                if line == "":
                    results = result_processor.process_result(result_lines)
                    for k, result in results.items():
                        if k not in k_results:
                            k_results[k] = {}
                        k_dict = k_results[k]
                        if model_name not in k_dict:
                            k_dict[model_name] = {}
                        k_dict[model_name][result_processor.dataset] = result

                    result_processor = None
                    result_lines = None

                else:
                    dataset = line[:-len(" evaluation")]
                    result_processor = get_dataset_processor(dataset)
                    result_lines = []

            elif result_processor is None:
                continue

            else:
                result_lines.append(line)

        for k, k_dict in k_results.items():
            if model_name in k_dict:
                for dataset in datasets:
                    assert dataset in k_dict[model_name], f"Missing {k}-shot {dataset} results for {model_name}"

    results_to_file(k_results, output_file, datasets)


if __name__=="__main__":
    args = parse_args()

    datasets = args.datasets.split(",")
    run_processing(args.input_path, args.output_file, datasets)

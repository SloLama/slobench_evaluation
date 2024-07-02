import operator
import warnings

import numpy as np
import scipy.stats as stats

from sklearn.metrics import f1_score, precision_score, recall_score
from Levenshtein import distance as lev_distance

SUPPORTED_CI_METHODS = [
    "std",
    "quantile_bootstrap"
]


def std_ci(values, alpha):
    standard_error = stats.sem(values)
    interval_range = standard_error * stats.t.ppf((1 + alpha) / 2, len(values) - 1)
    mean = np.mean(values)

    return mean - interval_range, mean + interval_range


def mean_quantile_bootstrap_ci(values, alpha, n_samples, seed):
    np.random.seed(seed)

    n = len(values)
    bootstrap_samples = []
    for _ in range(n_samples):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_samples.append(np.mean(sample))

    bootstrap_samples = np.array(bootstrap_samples)
    bootstrap_samples.sort()
    q_low = int(np.floor(n_samples * ((1 - alpha) / 2)))
    q_high = int(np.ceil(n_samples * ((1 + alpha) / 2)))

    return bootstrap_samples[q_low], bootstrap_samples[q_high]


def quantile_bootstrap_ci(metric, y_pred, y_true, alpha, n_samples, seed):
    np.random.seed(seed)

    n = len(y_pred)
    values = np.empty((n, 2))
    values[:, 0] = y_pred
    values[:, 1] = y_true
    bootstrap_samples = []
    for _ in range(n_samples):
        sample_idcs = np.random.choice(np.arange(n), size=n, replace=True)
        sample = values[sample_idcs, :]
        sample_result = metric(sample[:, 0], sample[:, 1])
        if isinstance(sample_result, np.ndarray):
            sample_result = sample_result.tolist()
        bootstrap_samples.append(sample_result)

    bootstrap_samples = np.array(bootstrap_samples)
    bootstrap_samples.sort(axis=0)
    q_low = int(np.floor(n_samples * ((1 - alpha) / 2)))
    q_high = int(np.ceil(n_samples * ((1 + alpha) / 2)))

    return bootstrap_samples[q_low], bootstrap_samples[q_high]


def accuracy(y_pred, y_true):
    return (y_pred == y_true).astype(int)


def weighted_similarity(y, majority_labels):
    weighted_averages = []
    for labels, weights in zip(y, majority_labels):
        if len(weights) < len(labels):
            new_weights = np.zeros(len(labels))
            new_weights[:len(weights)] = weights
            weights = new_weights

        positive_labels = labels == 1
        if np.sum(positive_labels) == 0:
            positive_similarity = 0
        else:
            positive_similarity = np.sum(weights[positive_labels]) / np.sum(positive_labels)

        negative_labels = labels == 0
        if np.sum(negative_labels) == 0:
            negative_similarity = 0
        else:
            negative_similarity = np.sum(weights[negative_labels]) / np.sum(negative_labels)

        weighted_averages.append(positive_similarity - negative_similarity)

    return np.array(weighted_averages)


def exact_match(y_pred, y_true):
    return np.array([np.all(pred == true) for pred, true in zip(y_pred, y_true)])


def per_instance_f1(y_pred, y_true):
    return np.array([f1_score(pred, truth) for pred, truth in zip(y_pred, y_true)])


def multiclass_f1(y_pred, y_true):
    return f1_score(y_true, y_pred, average="macro")


def per_class_f1(labels):
    def compute_f1(y_pred, y_true):
        return f1_score(y_true, y_pred, labels=labels, average=None)

    return compute_f1


def per_class_precision(labels):
    def compute_precision(y_pred, y_true):
        return precision_score(y_true, y_pred, labels=labels, average=None)

    return compute_precision


def per_class_recall(labels):
    def compute_recall(y_pred, y_true):
        return recall_score(y_true, y_pred, labels=labels, average=None)

    return compute_recall


def per_instance_lev_distance(y_pred, y_true):
    distances = []
    for pred, truth in zip(y_pred, y_true):
        pred_split = pred.split()
        truth_split = truth.split()

        # If truth is longer than prediction, check whether prediction is substring of truth, which is also ok
        n_candidates = len(truth_split) - len(pred_split) + 1
        candidates = [" ".join(truth_split[i:i+len(pred_split)]) for i in range(n_candidates)]
        instance_distance = [lev_distance(pred, candidate) for candidate in candidates]

        # Take the minimal distance among candidates as actual distance (the best matching susbstring)
        distances.append(min(instance_distance))

    return np.array(distances)


def lev_accuracy(y_pred, y_true):
    distances = per_instance_lev_distance(y_pred, y_true)
    n_words = [len(pred.split()) for pred in y_pred]

    return np.array([(distance / pred_len) <= 2.0 for distance, pred_len in zip(distances, n_words)])


class SloBenchEvaluator:
    def __init__(self, f_out) -> None:
        self.f_out = f_out

    def compute_general_stats(self, y_true):
        pass

    def evaluate(self, evaluation_params, predictions, true_labels, majority_labels=None, last_labels=None):
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(true_labels, np.ndarray):
            try:
                true_labels = np.array(true_labels)
            except:
                true_labels = np.array(true_labels, dtype=object)
        if (majority_labels is not None) and (not isinstance(majority_labels, np.ndarray)):
            majority_labels = np.array(majority_labels)
        if (last_labels is not None) and (not isinstance(last_labels, np.ndarray)):
            last_labels = np.array(last_labels)

        y_pred = self.transform_predictions(predictions, true_labels)

        y_pred, y_true, majority_labels, last_labels = self.filter_invalid_predictions(y_pred, true_labels, majority_labels, last_labels)

        if len(y_pred) == 0:
            self.f_out.write("There are no valid predictions. Evaluation can not be done.\n")
            return

        if evaluation_params.get("majority_correlation", False) and majority_labels is not None:
            self.compute_majority_correlation(y_pred, y_true, majority_labels, evaluation_params.get("ci", None))

        if evaluation_params.get("last_example_correlation", False) and last_labels is not None:
            self.compute_last_example_correlation(y_pred, y_true, last_labels, evaluation_params.get("ci", None))

        self.compute_model_loss(y_pred, y_true, evaluation_params.get("ci", None))

    def compute_metric(self, metric, y_pred, y_true, ci_params):
        result = metric(y_pred, y_true)

        assert ci_params["type"] == "quantile_bootstrap", f"{ci_params['type']} CI method is not supported for non-mean metrics. Please use quantile bootstrap instead."

        ci = quantile_bootstrap_ci(
            metric,
            y_pred,
            y_true,
            ci_params.get("alpha", 0.95),
            ci_params.get("bootstrap_samples", 1000),
            ci_params.get("seed", 42)
        )

        return result, ci

    def compute_mean_metric(self, metric, y_pred, y_true, ci_params):
        elementwise_results = metric(y_pred, y_true)

        if ci_params is None:
            ci = None
        else:
            ci_type = ci_params["type"]
            alpha = ci_params.get("alpha", 0.95)
            assert (
                    ci_type in SUPPORTED_CI_METHODS
            ), f"Unsupported CI computation method {ci_type}. Supported CI computation methods: {SUPPORTED_CI_METHODS}"

            if ci_type == "std":
                ci = std_ci(elementwise_results, alpha)
            elif ci_type == "quantile_bootstrap":
                ci = mean_quantile_bootstrap_ci(
                    elementwise_results,
                    alpha,
                    ci_params.get("bootstrap_samples", 1000),
                    ci_params.get("seed", 42)
                )

        return np.mean(elementwise_results), ci

    def transform_predictions(self, predictions, true_labels):
        pass

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        pass

    def compute_majority_correlation(self, y_pred, y_true, majority_labels, ci_params):
        pass

    def compute_last_example_correlation(self, y_pred, y_true, last_labels, ci_params):
        pass

    def compute_model_loss(self, y_pred, y_true, ci_params):
        pass


class BoolQEvaluator(SloBenchEvaluator):
    def __init__(self, f_out) -> None:
        super().__init__(f_out)

    def compute_general_stats(self, y_true):
        n_instances = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_instances - n_positive

        self.f_out.write("BoolQ evaluation set stats:\n")
        self.f_out.write(f"Number of instances: {n_instances}\n")
        self.f_out.write(f"Number of positive instances: {n_positive} ({100 * (n_positive / n_instances):.2f} %)\n")
        self.f_out.write(f"Number of negative instances: {n_negative} ({100 * (n_negative / n_instances):.2f} %)\n")

    def transform_predictions(self, predictions, true_labels):
        def transform_prediction(pred):
            pred = pred.strip()

            if len(pred) > 3:
                pred = pred[:3]

            if pred.lower() == "da." or pred.lower() == "da":
                return True

            if pred.lower() == "ne." or pred.lower() == "ne":
                return False

            return None

        return np.array(list(map(transform_prediction, predictions)))

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        valid_predictions = [pred is not None for pred in y_pred]

        n_invalid = len(y_pred) - sum(valid_predictions)
        self.f_out.write(f"Number of invalid predictions: {n_invalid} ({100 * (n_invalid / len(y_pred)):.2f} %)\n")

        if majority_labels is not None:
            majority_labels = majority_labels[valid_predictions]
        if last_labels is not None:
            last_labels = last_labels[valid_predictions]

        return y_pred[valid_predictions], y_true[valid_predictions], majority_labels, last_labels

    def compute_majority_correlation(self, y_pred, y_true, majority_labels, ci_params):
        has_majority = [label is not None for label in majority_labels]
        n_majority = sum(has_majority)
        self.f_out.write(f"Number of examples containing majority label: {n_majority} ({100 * (n_majority / len(y_pred))} %)\n")

        y_pred = y_pred[has_majority]
        majority_labels = majority_labels[has_majority]

        cor, ci, = self.compute_mean_metric(accuracy, y_pred, majority_labels, ci_params)

        output = f"Percentage of repsonses equal to majority label: {100 * cor:.2f} %"
        if ci is None:
            output += "\n"
        else:
            output += f" [{100 * ci[0]:.2f} %, {100 * ci[1]:.2f} %]\n"

        self.f_out.write(output)

    def compute_last_example_correlation(self, y_pred, y_true, last_labels, ci_params):
        cor, ci, = self.compute_mean_metric(accuracy, y_pred, last_labels, ci_params)

        output = f"Percentage of repsonses equal to the label of last example: {100 * cor:.2f} %"
        if ci is None:
            output += "\n"
        else:
            output += f" [{100 * ci[0]:.2f} %, {100 * ci[1]:.2f} %]\n"

        self.f_out.write(output)

    def compute_model_loss(self, y_pred, y_true, ci_params):
        loss, ci = self.compute_mean_metric(accuracy, y_pred, y_true, ci_params)

        output = f"Model's accuracy: {loss:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0]:.4f}, {ci[1]:.4f}]\n"

        self.f_out.write(output)


class MultiRCEvaluator(SloBenchEvaluator):
    def __init__(self, f_out):
        super().__init__(f_out)

    def compute_general_stats(self, y_true):
        self.f_out.write("MultiRC evaluation set stats:\n")

        n_answers = list(map(len, y_true))
        max_answers = max(n_answers)
        self.f_out.write(f"Minimal number of answers for a question: {min(n_answers)}\n")
        self.f_out.write(f"Maximal number of answers for a question: {max_answers}\n")
        self.f_out.write(f"Average number of answers per question: {np.mean(n_answers):.2f}\n")

        positive_answers = np.zeros(max_answers, dtype=int)
        missing_answers = np.zeros(max_answers, dtype=int)

        for answer_labels in y_true:
            q_answers = len(answer_labels)
            for i, label in enumerate(answer_labels):
                if label == 1:
                    positive_answers[i] += 1

            for j in range(q_answers, max_answers):
                missing_answers[j] += 1

        n_examples = len(y_true)
        positive_percentages = positive_answers.astype(float) / n_examples
        missing_percentages = missing_answers.astype(float) / n_examples

        negative_answers = n_examples - positive_answers - missing_answers
        negative_percentages = 1.0 - positive_percentages - missing_percentages

        self.f_out.write("Per answer statistics:\n")
        for i in range(max_answers):
            self.f_out.write(f"Answer {i}: {positive_answers[i]} ({100*positive_percentages[i]:.2f} %) positive examples, {negative_answers[i]} ({100*negative_percentages[i]:.2f} %) negative examples, {missing_answers[i]} ({100*missing_percentages[i]:.2f} %) missing examples.\n")

    def transform_predictions(self, predictions, true_labels):
        def transform_prediction(example):
            pred, label = example

            # Avoid errors due to additional spaces at the beginning
            pred = pred.lstrip()

            # Remove sentences such as "Pravilni odgovori:", "Pravilni odgovori so:", etc. from the beginning
            if pred.lower().startswith("pravilni") and ":" in pred:
                pred = pred.split(":")[1].strip()

            # Avoid errors due to (lack) of spaces after commas
            pred = pred.replace(", ", ",")
            # Find valid part of the prediction
            valid_chars = [str(i) for i in range(10)] + [","]
            valid_idx = 0
            for c in pred:
                if c in valid_chars:
                    valid_idx += 1
                else:
                    break

            if valid_idx == 0:
                return None

            pred = pred[:valid_idx]

            answers = pred.split(",")
            answers = [int(answer) for answer in answers if answer != ""]
            if answers == []:
                return None

            transformed_pred = np.zeros(max(len(label), max(answers) + 1), dtype=int)
            for answer in answers:
                transformed_pred[answer] = 1

            return transformed_pred

        return np.array(list(map(transform_prediction, zip(predictions, true_labels))), dtype=object)

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        valid_predictions = [pred is not None for pred in y_pred]

        n_invalid = len(y_pred) - sum(valid_predictions)
        self.f_out.write(f"Number of invalid predictions: {n_invalid} ({100 * (n_invalid / len(y_pred)):.2f} %)\n")

        if majority_labels is not None:
            majority_labels = majority_labels[valid_predictions]
        if last_labels is not None:
            last_labels = last_labels[valid_predictions]

        y_pred = y_pred[valid_predictions]
        y_true = y_true[valid_predictions]

        # Add possible answers to true labels, if model predicted the answer number higher than the number of answers
        for i, (pred, label) in enumerate(zip(y_pred, y_true)):
            if len(pred) > len(label):
                new_label = np.zeros(len(pred), dtype=int)
                new_label[:len(label)] = label
                y_true[i] = new_label

        # Add possible answers to majority labels
        if majority_labels is not None:
            for i, (pred, label) in enumerate(zip(y_pred, majority_labels)):
                if len(pred) > len(label):
                    new_label = np.zeros(len(pred))
                    new_label[:len(label)] = label
                    majority_labels[i] = new_label

        # Add possible answers to last labels
        if last_labels is not None:
            for i, (pred, label) in enumerate(zip(y_pred, last_labels)):
                if len(pred) > len(label):
                    new_label = np.zeros(len(pred))
                    new_label[:len(label)] = label
                    last_labels[i] = new_label

        return y_pred, y_true, majority_labels, last_labels

    def compute_majority_correlation(self, y_pred, y_true, majority_labels, ci_params):
        # Transform predictions to match the majority label size
        new_preds = []
        for pred, label in zip(y_pred, majority_labels):
            new_pred = np.zeros(max(len(pred), len(label)), dtype=int)
            new_pred[:len(pred)] = pred
            new_preds.append(new_pred)
        # Compute similarity of predictions and majority labels
        pred_cor, pred_ci = self.compute_mean_metric(
            weighted_similarity,
            np.array(new_preds, dtype=object),
            majority_labels,
            ci_params.get("correlation", None)
        )

        # Transform true labels to match the majority label size
        new_labels = []
        for truth, majority in zip(y_true, majority_labels):
            new_label = np.zeros(max(len(truth), len(majority)), dtype=int)
            new_label[:len(truth)] = truth
            new_labels.append(new_label)
        # Compute similarity of true and majority labels
        true_cor, true_ci = self.compute_mean_metric(
            weighted_similarity,
            np.array(new_labels, dtype=object),
            majority_labels,
            ci_params.get("correlation", None)
        )

        output = f"Weighted similarity of predicted labels to example labels: {pred_cor:.4f}"
        if pred_ci is None:
            output += "\n"
        else:
            output += f" [{pred_ci[0]:.4f}, {pred_ci[1]:.4f}]\n"
        self.f_out.write(output)

        output = f"Weighted similarity of true labels to example labels: {true_cor:.4f}"
        if true_ci is None:
            output += "\n"
        else:
            output += f" [{true_ci[0]:.4f}, {true_ci[1]:.4f}]\n"
        self.f_out.write(output)

        self.f_out.write(f"Difference between similarity of predicted and true labels: {pred_cor-true_cor:.4f}\n")

    def compute_last_example_correlation(self, y_pred, y_true, last_labels, ci_params):
        # Transform predictions to match the last label size
        new_preds = []
        for pred, label in zip(y_pred, last_labels):
            new_pred = np.zeros(max(len(pred), len(label)), dtype=int)
            new_pred[:len(pred)] = pred
            new_preds.append(new_pred)
        # Compute similarity of predictions and majority labels
        pred_cor, pred_ci = self.compute_mean_metric(
            weighted_similarity,
            np.array(new_preds, dtype=object),
            last_labels,
            ci_params.get("correlation", None)
        )

        # Transform true labels to match the last label size
        new_labels = []
        for truth, last in zip(y_true, last_labels):
            new_label = np.zeros(max(len(truth), len(last)), dtype=int)
            new_label[:len(truth)] = truth
            new_labels.append(new_label)
        # Compute similarity of true and majority labels
        true_cor, true_ci = self.compute_mean_metric(
            weighted_similarity,
            np.array(new_labels, dtype=object),
            last_labels,
            ci_params.get("correlation", None)
        )

        output = f"Weighted similarity of predicted labels to last example label: {pred_cor:.4f}"
        if pred_ci is None:
            output += "\n"
        else:
            output += f" [{pred_ci[0]:.4f}, {pred_ci[1]:.4f}]\n"
        self.f_out.write(output)

        output = f"Weighted similarity of true labels to last example label: {true_cor:.4f}"
        if true_ci is None:
            output += "\n"
        else:
            output += f" [{true_ci[0]:.4f}, {true_ci[1]:.4f}]\n"
        self.f_out.write(output)

        self.f_out.write(f"Difference between similarity of predicted and true labels: {pred_cor - true_cor:.4f}\n")

    def compute_model_loss(self, y_pred, y_true, ci_params):
        # Exact match
        em, em_ci = self.compute_mean_metric(exact_match, y_pred, y_true, ci_params.get("exact_match", None))
        output = f"Exact match: {em:.4f}"
        if em_ci is None:
            output += "\n"
        else:
            output += f" [{em_ci[0]:.4f}, {em_ci[1]:.4f}]\n"
        self.f_out.write(output)

        # Per question F1
        f1_q, f1_q_ci = self.compute_mean_metric(per_instance_f1, y_pred, y_true, ci_params.get("per_question_f1", None))
        output = f"Per question F1-score: {f1_q:.4f}"
        if f1_q_ci is None:
            output += "\n"
        else:
            output += f" [{f1_q_ci[0]:.4f}, {f1_q_ci[1]:.4f}]\n"
        self.f_out.write(output)

        # F1 over all answers
        reshaped_preds, reshaped_true = np.array([], dtype=int), np.array([], dtype=int)
        for pred, truth in zip(y_pred, y_true):
            reshaped_preds = np.concatenate([reshaped_preds, pred])
            reshaped_true = np.concatenate([reshaped_true, truth])
        f1_a, f1_a_ci = self.compute_metric(f1_score, reshaped_preds, reshaped_true, ci_params.get("all_answers_f1", None))
        output = f"F1-score over all answers: {f1_a:.4f}"
        if f1_a_ci is None:
            output += "\n"
        else:
            output += f" [{f1_a_ci[0]:.4f}, {f1_a_ci[1]:.4f}]\n"
        self.f_out.write(output)


class WSCEvaluator(BoolQEvaluator):
    def __init__(self, f_out):
        super().__init__(f_out)

    def compute_general_stats(self, y_true):
        n_instances = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_instances - n_positive

        self.f_out.write("WSC evaluation set stats:\n")
        self.f_out.write(f"Number of instances: {n_instances}\n")
        self.f_out.write(f"Number of positive instances: {n_positive} ({100 * (n_positive / n_instances):.2f} %)\n")
        self.f_out.write(f"Number of negative instances: {n_negative} ({100 * (n_negative / n_instances):.2f} %)\n")


class WSCGenerativeEvaluator(SloBenchEvaluator):
    def __init__(self, f_out):
        super().__init__(f_out)

    def transform_predictions(self, predictions, true_labels):
        def transform_prediction(example):
            pred, truth = example

            pred = pred.strip().lower()
            if pred == "":
                return None

            pred = pred.split()
            truth = truth.split()
            if len(pred) > len(truth):
                pred = pred[:len(truth)]

            return " ".join(pred)

        return np.array(list(map(transform_prediction, zip(predictions, true_labels))))

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        valid_predictions = [pred is not None for pred in y_pred]

        n_invalid = len(y_pred) - sum(valid_predictions)
        self.f_out.write(f"Number of invalid predictions: {n_invalid} ({100 * (n_invalid / len(y_pred)):.2f} %)\n")

        y_true = np.array([truth.lower() for truth in y_true])
        y_true = y_true[valid_predictions]

        if last_labels is not None:
            last_labels = np.array([label.lower() for label in last_labels])
            last_labels = last_labels[valid_predictions]

        return y_pred[valid_predictions], y_true, None, last_labels

    def compute_last_example_correlation(self, y_pred, y_true, last_labels, ci_params):
        split_preds = [pred.split() for pred in y_pred]
        split_labels = [label.split() for label in last_labels]
        y_pred = np.array([" ".join(pred[:min(len(pred), len(label))]) for pred, label in zip(split_preds, split_labels)])
        pred_cor, pred_ci = self.compute_mean_metric(per_instance_lev_distance, y_pred, last_labels, ci_params)

        output = f"Average Levenshtein distance between model's predictions and labels of last example: {pred_cor:.2f}"
        if pred_ci is None:
            output += "\n"
        else:
            output += f" [{pred_ci[0]:.2f}, {pred_ci[1]:.2f}]\n"

        self.f_out.write(output)

        split_truths = [truth.split() for truth in y_true]
        y_true = np.array([" ".join(truth[:min(len(truth), len(label))]) for truth, label in zip(split_truths, split_labels)])
        true_cor, true_ci = self.compute_mean_metric(per_instance_lev_distance, y_true, last_labels, ci_params)

        output = f"Average Levenshtein distance between true labels and labels of last example: {true_cor:.2f}"
        if true_ci is None:
            output += "\n"
        else:
            output += f" [{true_ci[0]:.2f}, {true_ci[1]:.2f}]\n"

        self.f_out.write(output)

        self.f_out.write(f"Difference between distance of true and predicted labels: {true_cor - pred_cor:.2f}\n")

    def compute_model_loss(self, y_pred, y_true, ci_params):
        loss, ci = self.compute_mean_metric(per_instance_lev_distance, y_pred, y_true, ci_params)

        output = f"Average Levenshtein distance between model's predictions and true labels: {loss:.2f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0]:.2f}, {ci[1]:.2f}]\n"

        self.f_out.write(output)

        acc, acc_ci = self.compute_mean_metric(lev_accuracy, y_pred, y_true, ci_params)

        output = f"Model's accuracy (based on Levenshtein distance): {100 * acc:.2f} %"
        if acc_ci is None:
            output += "\n"
        else:
            output += f" [{100 * acc_ci[0]:.2f} %, {100 * acc_ci[1]:.2f} %]\n"

        self.f_out.write(output)


class COPAEvaluator(BoolQEvaluator):
    def compute_general_stats(self, y_true):
        n_instances = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_instances - n_positive

        self.f_out.write("COPA evaluation set stats:\n")
        self.f_out.write(f"Number of instances: {n_instances}\n")
        self.f_out.write(f"Number of instances where choice 1 is correct: {n_negative} ({100 * (n_negative / n_instances):.2f} %)\n")
        self.f_out.write(f"Number of instances where choice 2 is correct: {n_positive} ({100 * (n_positive / n_instances):.2f} %)\n")

    def transform_predictions(self, predictions, true_labels):
        def transform_prediction(pred):
            pred = pred.strip()

            if len(pred) > 1:
                pred = pred[:1]

            if pred == "1":
                return 0

            if pred == "2":
                return 1

            return None

        return np.array(list(map(transform_prediction, predictions)))


class RTEEvaluator(BoolQEvaluator):
    def compute_general_stats(self, y_true):
        n_instances = len(y_true)
        n_positive = np.sum(y_true)
        n_negative = n_instances - n_positive

        self.f_out.write("RTE evaluation set stats:\n")
        self.f_out.write(f"Number of instances: {n_instances}\n")
        self.f_out.write(f"Number of positive instances: {n_positive} ({100 * (n_positive / n_instances):.2f} %)\n")
        self.f_out.write(f"Number of negative instances: {n_negative} ({100 * (n_negative / n_instances):.2f} %)\n")


class CBEvaluator(BoolQEvaluator):
    def compute_general_stats(self, y_true):
        n_instances = len(y_true)
        n_entailment = np.sum(y_true == 0)
        n_contradiction = np.sum(y_true == 1)
        n_neutral = n_instances - n_entailment - n_contradiction

        self.f_out.write("CB evaluation set stats:\n")
        self.f_out.write(f"Number of instances: {n_instances}\n")
        self.f_out.write(
            f"Number of entailment instances: {n_entailment} ({100 * (n_entailment / n_instances):.2f} %)\n")
        self.f_out.write(
            f"Number of contradiction instances: {n_contradiction} ({100 * (n_contradiction / n_instances):.2f} %)\n")
        self.f_out.write(
            f"Number of neutral instances: {n_neutral} ({100 * (n_neutral / n_instances):.2f} %)\n")

    def transform_predictions(self, predictions, true_labels):
        def transform_prediction(pred):
            if pred[:5].lower() == "drži." or pred.lower() == "drži":
                return 0
            if pred[:8].lower() == "ne drži." or pred.lower() == "ne drži":
                return 1
            if pred[:8].lower() == "ne vemo." or pred.lower() == "ne vemo":
                return 2

            return None

        return np.array(list(map(transform_prediction, predictions)))

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        valid_predictions = [pred is not None for pred in y_pred]

        n_invalid = len(y_pred) - sum(valid_predictions)
        self.f_out.write(f"Number of invalid predictions: {n_invalid} ({100 * (n_invalid / len(y_pred)):.2f} %)\n")

        if majority_labels is not None:
            majority_labels = majority_labels[valid_predictions]
        if last_labels is not None:
            last_labels = last_labels[valid_predictions]

        return y_pred[valid_predictions].astype(int), y_true[valid_predictions], majority_labels, last_labels

    def compute_majority_correlation(self, y_pred, y_true, majority_labels, ci_params):
        has_majority = [label is not None for label in majority_labels]
        n_majority = sum(has_majority)
        self.f_out.write(f"Number of examples containing majority label: {n_majority} ({100 * (n_majority / len(y_pred))} %)\n")

        y_pred = y_pred[has_majority]
        majority_labels = majority_labels[has_majority]

        cor, ci, = self.compute_mean_metric(accuracy, y_pred, majority_labels, ci_params.get("correlation", None))

        output = f"Percentage of repsonses equal to majority label: {100 * cor:.2f} %"
        if ci is None:
            output += "\n"
        else:
            output += f" [{100 * ci[0]:.2f} %, {100 * ci[1]:.2f} %]\n"

        self.f_out.write(output)

    def compute_last_example_correlation(self, y_pred, y_true, last_labels, ci_params):
        cor, ci, = self.compute_mean_metric(accuracy, y_pred, last_labels, ci_params.get("correlation", None))

        output = f"Percentage of repsonses equal to the label of last example: {100 * cor:.2f} %"
        if ci is None:
            output += "\n"
        else:
            output += f" [{100 * ci[0]:.2f} %, {100 * ci[1]:.2f} %]\n"

        self.f_out.write(output)

    def compute_model_loss(self, y_pred, y_true, ci_params):
        loss, ci = self.compute_mean_metric(accuracy, y_pred, y_true, ci_params.get("accuracy", None))

        output = f"Model's accuracy: {loss:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0]:.4f}, {ci[1]:.4f}]\n"

        self.f_out.write(output)

        f1_loss, f1_ci = self.compute_metric(multiclass_f1, y_pred, y_true, ci_params.get("f1", None))
        output = f"Average F1-score over labels: {f1_loss:.4f}"
        if f1_ci is None:
            output += "\n"
        else:
            output += f" [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]\n"

        self.f_out.write(output)


class NLIEvaluator(CBEvaluator):
    def compute_general_stats(self, y_true):
        n_instances = len(y_true)
        n_entailment = np.sum(y_true == 1)
        n_contradiction = np.sum(y_true == 0)
        n_neutral = n_instances - n_entailment - n_contradiction

        self.f_out.write("NLI evaluation set stats:\n")
        self.f_out.write(f"Number of instances: {n_instances}\n")
        self.f_out.write(
            f"Number of entailment instances: {n_entailment} ({100 * (n_entailment / n_instances):.2f} %)\n")
        self.f_out.write(
            f"Number of contradiction instances: {n_contradiction} ({100 * (n_contradiction / n_instances):.2f} %)\n")
        self.f_out.write(
            f"Number of neutral instances: {n_neutral} ({100 * (n_neutral / n_instances):.2f} %)\n")

    def transform_predictions(self, predictions, true_labels):
        def transform_prediction(pred):
            if pred[:9].lower() == "sosledje." or pred.lower() == "sosledje":
                return 1
            if pred[:14].lower() == "nasprotovanje." or pred.lower() == "nasprotovanje":
                return 0
            if pred[:12].lower() == "nevtralnost." or pred.lower() == "nevtralnost":
                return 2

            return None

        return np.array(list(map(transform_prediction, predictions)))

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        valid_predictions = [pred is not None for pred in y_pred]

        n_invalid = len(y_pred) - sum(valid_predictions)
        self.f_out.write(f"Number of invalid predictions: {n_invalid} ({100 * (n_invalid / len(y_pred)):.2f} %)\n")

        if majority_labels is not None:
            majority_labels = majority_labels[valid_predictions]
        if last_labels is not None:
            last_labels = last_labels[valid_predictions]

        return y_pred[valid_predictions].astype(int), y_true[valid_predictions], majority_labels, last_labels

    def compute_model_loss(self, y_pred, y_true, ci_params):
        labels = np.array([0, 1, 2])

        warnings.filterwarnings("ignore")

        loss, ci = self.compute_mean_metric(accuracy, y_pred, y_true, ci_params.get("accuracy", None))

        output = f"Model's accuracy: {loss:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0]:.4f}, {ci[1]:.4f}]\n"

        self.f_out.write(output)

        precision_loss, precision_ci = self.compute_metric(
            per_class_precision(labels), y_pred, y_true, ci_params.get("precision", None)
        )
        self.print_metric_results("precision", precision_loss, precision_ci)

        recall_loss, recall_ci = self.compute_metric(
            per_class_recall(labels), y_pred, y_true, ci_params.get("recall", None)
        )
        self.print_metric_results("recall", recall_loss, recall_ci)

        f1_loss, f1_ci = self.compute_metric(
            per_class_f1(labels), y_pred, y_true, ci_params.get("f1", None)
        )
        self.print_metric_results("F1", f1_loss, f1_ci)

        warnings.filterwarnings("default")

    def print_metric_results(self, metric_name, result, ci):
        output = f"Entailment {metric_name}: {result[1]:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0][1]:.4f}, {ci[1][1]:.4f}]\n"
        output += f"Contradiction {metric_name}: {result[0]:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0][0]:.4f}, {ci[1][0]:.4f}]\n"
        output += f"Neutral {metric_name}: {result[2]:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0][2]:.4f}, {ci[1][2]:.4f}]\n"

        self.f_out.write(output)

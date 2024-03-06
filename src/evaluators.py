import numpy as np
import scipy.stats as stats

SUPPORTED_CI_METHODS = [
    "std",
    "quantile_bootstrap"
]


def std_ci(values, alpha):
    standard_error = stats.sem(values)
    interval_range = standard_error * stats.t.ppf((1 + alpha) / 2, len(values) - 1)
    mean = np.mean(values)

    return mean - interval_range, mean + interval_range


def quantile_bootstrap_ci(values, alpha, n_samples, seed):
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


def accuracy(y_pred, y_true):
    return (y_pred == y_true).astype(int)


class SloBenchEvaluator:
    def __init__(self, f_out) -> None:
        self.f_out = f_out

    def compute_general_stats(self, y_true):
        pass

    def evaluate(self, evaluation_params, predictions, true_labels, majority_labels=None, last_labels=None):
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(true_labels, np.ndarray):
            true_labels = np.array(true_labels)
        if (majority_labels is not None) and (not isinstance(majority_labels, np.ndarray)):
            majority_labels = np.array(majority_labels)
        if (last_labels is not None) and (not isinstance(last_labels, np.ndarray)):
            last_labels = np.array(last_labels)

        y_pred = self.transform_predictions(predictions)

        y_pred, y_true, majority_labels, last_labels = self.filter_invalid_predictions(y_pred, true_labels, majority_labels, last_labels)

        if len(y_pred) == 0:
            self.f_out.write("There are no valid predictions. Evaluation can not be done.\n")
            return

        if evaluation_params.get("majority_correlation", False) and majority_labels is not None:
            self.compute_majority_correlation(y_pred, majority_labels, evaluation_params.get("ci", None))

        if evaluation_params.get("last_example_correlation", False) and last_labels is not None:
            self.compute_last_example_correlation(y_pred, last_labels, evaluation_params.get("ci", None))

        self.compute_model_loss(y_pred, y_true, evaluation_params.get("ci", None))

    def compute_metric(self, metric, y1, y2, ci_params):
        elementwise_results = metric(y1, y2)

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
                ci = quantile_bootstrap_ci(
                    elementwise_results,
                    alpha,
                    ci_params.get("bootstrap_samples", 1000),
                    ci_params.get("seed", 42)
                )

        return np.mean(elementwise_results), ci

    def transform_predictions(self, predictions):
        pass

    def filter_invalid_predictions(self, y_pred, y_true, majority_labels, last_labels):
        pass

    def compute_majority_correlation(self, y_pred, majority_labels, ci_params):
        pass

    def compute_last_example_correlation(self, y_pred, last_labels, ci_params):
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

    def transform_predictions(self, predictions):
        def transform_prediction(pred):
            if len(pred) > 2:
                pred = pred[-3:]

            if pred.lower() == "da.":
                return True

            if pred.lower() == "ne.":
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

    def compute_majority_correlation(self, y_pred, majority_labels, ci_params):
        has_majority = [label is not None for label in majority_labels]
        n_majority = sum(has_majority)
        self.f_out.write(f"Number of examples containing majority label: {n_majority} ({100 * (n_majority / len(y_pred))} %)\n")

        y_pred = y_pred[has_majority]
        majority_labels = majority_labels[has_majority]

        cor, ci, = self.compute_metric(accuracy, y_pred, majority_labels, ci_params)

        output = f"Percentage of repsonses equal to majority label: {100 * cor:.2f} %"
        if ci is None:
            output += "\n"
        else:
            output += f" [{100 * ci[0]:.2f} %, {100 * ci[1]:.2f}]\n"

        self.f_out.write(output)

    def compute_last_example_correlation(self, y_pred, last_labels, ci_params):
        cor, ci, = self.compute_metric(accuracy, y_pred, last_labels, ci_params)

        output = f"Percentage of repsonses equal to the label of last example: {100 * cor:.2f} %"
        if ci is None:
            output += "\n"
        else:
            output += f" [{100 * ci[0]:.2f} %, {100 * ci[1]:.2f}]\n"

        self.f_out.write(output)

    def compute_model_loss(self, y_pred, y_true, ci_params):
        loss, ci = self.compute_metric(accuracy, y_pred, y_true, ci_params)

        output = f"Model's accuracy: {loss:.4f}"
        if ci is None:
            output += "\n"
        else:
            output += f" [{ci[0]:.4f}, {ci[1]:.4f}]\n"

        self.f_out.write(output)
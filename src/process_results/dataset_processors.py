class DatasetProcessor:
    def __init__(self) -> None:
        self.dataset = None

    def process_result(self, lines):
        results = {}

        k = None
        result_lines = None
        for line in lines:
            if line.startswith("Results for"):
                k_string = line.split()[2][:-len("-shot")]
                k = int(k_string)
                result_lines = []

            elif k is None:
                continue

            elif line == "":
                if result_lines is not None:
                    result = self.extract_result(result_lines)
                    results[k] = result

                k = None
                result_lines = None

            elif line.strip("=") == "":
                continue

            else:
                result_lines.append(line)

        # Add the last results
        if result_lines is not None:
            result = self.extract_result(result_lines)
            results.append(result)

        return results

    def extract_result(self, lines):
        pass

    def round_number(self, number, ndigits=2):
        if isinstance(number, str):
            number = float(number)
        rounded = round(number, ndigits=ndigits)

        return str(rounded)
    
    def get_invalid_predictions(self, invalid_predictions_line):
        invalid_predictions = invalid_predictions_line.split()[-2]
        invalid_predictions = invalid_predictions[len("("):] + "%"

        return invalid_predictions
    
    def get_metric(self, metric_line):
        metric_line = metric_line.split()
        metric_value = metric_line[-3]
        metric_value = self.round_number(metric_value)
        metric_ci_l = metric_line[-2][len("["):][:-len(",")]
        metric_ci_l = self.round_number(metric_ci_l)
        metric_ci_r = metric_line[-1][:-len("]")]
        metric_ci_r = self.round_number(metric_ci_r)
        metric = f"{metric_value} [{metric_ci_l}, {metric_ci_r}]"

        return metric


class BoolQProcessor(DatasetProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "BoolQ"

    def extract_result(self, lines):
        # Get invalid predictions
        invalid_predictions_line = lines[0]
        invalid_predictions = self.get_invalid_predictions(invalid_predictions_line)

        if lines[-1] == "There are no valid predictions. Evaluation can not be done.":
            results_dict = {
                "accuracy": "/",
                "invalid_predictions": invalid_predictions,
                "majority_correlation": "/",
                "last_example_correlation": "/"
            }

            return results_dict

        # Get accuracy
        accuracy_line = lines[-1]
        accuracy = self.get_metric(accuracy_line)

        majority_correlation, last_example_correlation = "/", "/"

        # Get majority correlation
        if len(lines) > 2:
            majority_correlation_line = lines[2]
            majority_correlation = self.get_correlation(majority_correlation_line)

        # Get last example correlation
        if len(lines) > 4:
            last_example_correlation_line = lines[3]
            last_example_correlation = self.get_correlation(last_example_correlation_line)

        # Case when k=1 meaning that majority correlation is last example correlation
        if last_example_correlation == "/" and majority_correlation != "/":
            last_example_correlation, majority_correlation = majority_correlation, last_example_correlation

        results_dict = {
            "accuracy": accuracy,
            "invalid_predictions": invalid_predictions,
            "majority_correlation": majority_correlation,
            "last_example_correlation": last_example_correlation
        }

        return results_dict


    def get_correlation(self, correlation_line):
        line_split = correlation_line.split(": ")
        correlation = line_split[1].split()[0]
        correlation = float(correlation) / 100

        return self.round_number(correlation)


class MultiRCProcessor(DatasetProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "MultiRC"

    def extract_result(self, lines):
        # Get invalid predictions
        invalid_predictions_line = lines[0]
        invalid_predictions = self.get_invalid_predictions(invalid_predictions_line)

        if lines[-1] == "There are no valid predictions. Evaluation can not be done.":
            results_dict = {
                "exact_match": "/",
                "per_question_f1": "/",
                "f1_over_all_answers": "/",
                "invalid_predictions": invalid_predictions,
                "majority_correlation": "/",
                "last_example_correlation": "/"
            }

            return results_dict

        # Get Exact match
        em_line = lines[-3]
        exact_match = self.get_metric(em_line)

        # Get per-question F1
        per_q_f1_line = lines[-2]
        per_question_f1 = self.get_metric(per_q_f1_line)

        # Get F1 over all answers
        f1_answers_line = lines[-1]
        f1_over_all_answers = self.get_metric(f1_answers_line)

        majority_correlation, last_example_correlation = "/", "/"

        # Get majority correlation
        if len(lines) > 4:
            majority_correlation_line = lines[3]
            majority_correlation = self.get_correlation(majority_correlation_line)

        # Get last example correlation
        if len(lines) > 7:
            last_example_correlation_line = lines[6]
            last_example_correlation = self.get_correlation(last_example_correlation_line)

        # Case when k=1 meaning that majority correlation is last example correlation
        if last_example_correlation == "/" and majority_correlation != "/":
            last_example_correlation, majority_correlation = majority_correlation, last_example_correlation

        results_dict = {
            "exact_match": exact_match,
            "per_question_f1": per_question_f1,
            "f1_over_all_answers": f1_over_all_answers,
            "invalid_predictions": invalid_predictions,
            "majority_correlation": majority_correlation,
            "last_example_correlation": last_example_correlation
        }

        return results_dict

    def get_correlation(self, correlation_line):
        line_split = correlation_line.split(": ")
        correlation = self.round_number(line_split[-1])

        return correlation


class WSCProcessor(BoolQProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "WSC"


class WSCGenerativeProcessor(DatasetProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "WSC_generative"

    def extract_result(self, lines):
        # Get invalid predictions
        invalid_predictions_line = lines[0]
        invalid_predictions = self.get_invalid_predictions(invalid_predictions_line)

        if lines[-1] == "There are no valid predictions. Evaluation can not be done.":
            results_dict = {
                "accuracy": "/",
                "levenshtein_distance": "/",
                "invalid_predictions": invalid_predictions
            }

            return results_dict
        
        # Get accuracy
        accuracy_line = lines[-1]
        accuracy_line = accuracy_line.split()
        accuracy_score = accuracy_line[-6]
        accuracy_score = float(accuracy_score) / 100
        accuracy_score = self.round_number(accuracy_score)
        accuracy_ci_l = accuracy_line[-4][len("["):]
        accuracy_ci_l = float(accuracy_ci_l) / 100
        accuracy_ci_l = self.round_number(accuracy_ci_l)
        accuracy_ci_r = accuracy_line[-2]
        accuracy_ci_r = float(accuracy_ci_r) / 100
        accuracy_ci_r = self.round_number(accuracy_ci_r)
        accuracy = f"{accuracy_score} [{accuracy_ci_l}, {accuracy_ci_r}]"

        # Get Levenhstein distance
        ld_line = lines[-2]
        levenhstein_distance = self.get_metric(ld_line)

        results_dict = {
            "accuracy": accuracy,
            "levenshtein_distance": levenhstein_distance,
            "invalid_predictions": invalid_predictions
        }

        return results_dict

    
class COPAProcessor(BoolQProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "COPA"


class RTEProcessor(BoolQProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "RTE"


class CBProcessor(BoolQProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "CB"

    def extract_result(self, lines):
        # Get invalid predictions
        invalid_predictions_line = lines[0]
        invalid_predictions = self.get_invalid_predictions(invalid_predictions_line)

        if lines[-1] == "There are no valid predictions. Evaluation can not be done.":
            results_dict = {
                "accuracy": "/",
                "f1": "/",
                "invalid_predictions": invalid_predictions,
                "majority_correlation": "/",
                "last_example_correlation": "/"
            }

            return results_dict
        
        # Get accuracy
        accuracy_line = lines[-2]
        accuracy = self.get_metric(accuracy_line)

        # Get F1
        f1_line = lines[-1]
        f1 = self.get_metric(f1_line)

        majority_correlation, last_example_correlation = "/", "/"

        # Get majority correlation
        if len(lines) > 3:
            majority_correlation_line = lines[2]
            majority_correlation = self.get_correlation(majority_correlation_line)

        # Get last example correlation
        if len(lines) > 5:
            last_example_correlation_line = lines[3]
            last_example_correlation = self.get_correlation(last_example_correlation_line)

        # Case when k=1 meaning that majority correlation is last example correlation
        if last_example_correlation == "/" and majority_correlation != "/":
            last_example_correlation, majority_correlation = majority_correlation, last_example_correlation

        results_dict = {
            "accuracy": accuracy,
            "f1": f1,
            "invalid_predictions": invalid_predictions,
            "majority_correlation": majority_correlation,
            "last_example_correlation": last_example_correlation
        }

        return results_dict


class NLIProcessor(BoolQProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = "NLI"

    def extract_result(self, lines):
        # Get invalid predictions
        invalid_predictions_line = lines[0]
        invalid_predictions = self.get_invalid_predictions(invalid_predictions_line)

        if lines[-1] == "There are no valid predictions. Evaluation can not be done.":
            results_dict = {
                "accuracy": "/",
                "precision_entailment": "/",
                "recall_entailment": "/",
                "f1_entailment": "/",
                "precision_neutral": "/",
                "recall_neutral": "/",
                "f1_neutral": "/",
                "precision_contradiction": "/",
                "recall_contradiction": "/",
                "f1_contradiction": "/",
                "invalid_predictions": invalid_predictions,
                "majority_correlation": "/",
                "last_example_correlation": "/"
            }

            return results_dict
        
        # Get accuracy
        accuracy_line = lines[-10]
        accuracy = self.get_metric(accuracy_line)
        
        # Get entailment metrics
        entailment_p_line = lines[-9]
        entailment_precision = self.get_metric(entailment_p_line)
        entailment_r_line = lines[-6]
        entailment_recall = self.get_metric(entailment_r_line)
        entailment_f1_line = lines[-3]
        entailment_f1 = self.get_metric(entailment_f1_line)

        # Get neutral metrics
        neutral_p_line = lines[-7]
        neutral_precision = self.get_metric(neutral_p_line)
        neutral_r_line = lines[-4]
        neutral_recall = self.get_metric(neutral_r_line)
        neutral_f1_line = lines[-1]
        neutral_f1 = self.get_metric(neutral_f1_line)

        # Get contradiction metrics
        contradiction_p_line = lines[-8]
        contradiction_precision = self.get_metric(contradiction_p_line)
        contradiction_r_line = lines[-5]
        contradiction_recall = self.get_metric(contradiction_r_line)
        contradiction_f1_line = lines[-2]
        contradiction_f1 = self.get_metric(contradiction_f1_line)

        majority_correlation, last_example_correlation = "/", "/"

        # Get majority correlation
        if len(lines) > 11:
            majority_correlation_line = lines[2]
            majority_correlation = self.get_correlation(majority_correlation_line)

        # Get last example correlation
        if len(lines) > 13:
            last_example_correlation_line = lines[3]
            last_example_correlation = self.get_correlation(last_example_correlation_line)

        # Case when k=1 meaning that majority correlation is last example correlation
        if last_example_correlation == "/" and majority_correlation != "/":
            last_example_correlation, majority_correlation = majority_correlation, last_example_correlation

        results_dict = {
            "accuracy": accuracy,
            "precision_entailment": entailment_precision,
            "recall_entailment": entailment_recall,
            "f1_entailment": entailment_f1,
            "precision_neutral": neutral_precision,
            "recall_neutral": neutral_recall,
            "f1_neutral": neutral_f1,
            "precision_contradiction": contradiction_precision,
            "recall_contradiction": contradiction_recall,
            "f1_contradiction": contradiction_f1,
            "invalid_predictions": invalid_predictions,
            "majority_correlation": majority_correlation,
            "last_example_correlation": last_example_correlation
        }

        return results_dict

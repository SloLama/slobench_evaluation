import json
import os
import random

import numpy as np

class SlobenchSubmissionCreator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.dataset = None

    def prepare_submission(self, predictions, data_info):
        predictions = self.transform_predictions(predictions)
        invalid_predictions = sum([pred is None for pred in predictions])
        print(f"Number of invalid predictions on {self.dataset} test set: {invalid_predictions} ({100 * invalid_predictions / len(predictions):.2f})")

        prediction_strings = self.write_predictions(predictions, data_info)

        file_ending = ".txt" if self.dataset == "NLI" else ".jsonl"
        f_out = open(os.path.join(self.output_dir, self.dataset + file_ending), "w")
        for line in prediction_strings:
            f_out.write(line + "\n")
        f_out.close()

    def transform_predictions(self, predictions):
        return []

    def write_predictions(self, predictions, data_info):
        output_lines = []
        for instance_info, pred in zip(data_info, predictions):
            instance_info["label"] = self.pred_to_string(pred)
            output_lines.append(json.dumps(instance_info, ensure_ascii=False))

        return output_lines

    def pred_to_string(self, pred):
        return None

    def get_data_info(self, instance):
        pass


class BoolQSubmissionCreator(SlobenchSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "BoolQ"

    def transform_predictions(self, predictions):
        def transform_prediction(pred):
            pred = pred.strip()

            if len(pred) > 3:
                pred = pred[:3]

            if pred.lower() in ["da.", "da", "da,"]:
                return True

            if pred.lower() in ["ne.", "ne", "ne,"]:
                return False

            return None

        return list(map(transform_prediction, predictions))

    def pred_to_string(self, pred):
        if pred is None:
            return random.choice(["true", "false"])
        if pred:
            return "true"
        return "false"

    def get_data_info(self, instance):
        return {"idx": int(instance["idx"])}


class CBSubmissionCreator(SlobenchSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "CB"

    def transform_predictions(self, predictions):
        def transform_prediction(pred):
            pred = pred.strip()

            if pred[:4].lower() == "drži":
                return 0
            if pred[:7].lower() == "ne drži":
                return 1
            if pred[:7].lower() == "ne vemo":
                return 2

            return None

        return list(map(transform_prediction, predictions))

    def pred_to_string(self, pred):
        if pred is None:
            return random.choice(["entailment", "contradiction", "neutral"])
        if pred == 0:
            return "entailment"
        if pred == 1:
            return "contradiction"
        return "neutral"

    def get_data_info(self, instance):
        return {"idx": int(instance["idx"])}


class COPASubmissionCreator(SlobenchSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "COPA"

    def transform_predictions(self, predictions):
        def transform_prediction(pred):
            pred = pred.strip()

            # Remove "hipoteza" from the start of the prediction
            pred = pred.lower()
            pred = pred.lstrip("hipoteza")
            pred = pred.lstrip()

            if len(pred) > 1:
                pred = pred[:1]

            if pred == "1":
                return 0

            if pred == "2":
                return 1

            return None

        return list(map(transform_prediction, predictions))

    def pred_to_string(self, pred):
        if pred is None:
            return random.choice([0, 1])

        return pred

    def get_data_info(self, instance):
        return {"idx": int(instance["idx"])}


class MultiRCSubmissionCreator(SlobenchSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "MultiRC"

    def prepare_submission(self, predictions, data_info):
        answer_idx = [instance["answer_idx"] for instance in data_info]
        predictions = self.transform_predictions(predictions, answer_idx)
        invalid_predictions = sum([pred is None for pred in predictions])
        print(f"Number of invalid predictions on {self.dataset} test set: {invalid_predictions} ({100 * invalid_predictions / len(predictions):.2f})")

        prediction_strings = self.write_predictions(predictions, data_info)

        file_ending = ".jsonl"
        f_out = open(os.path.join(self.output_dir, self.dataset + file_ending), "w")
        for line in prediction_strings:
            f_out.write(line + "\n")
        f_out.close()

    def transform_predictions(self, predictions, answer_idx):
        def transform_prediction(example):
            pred, answers = example

            # Avoid errors due to additional spaces at the beginning
            pred = pred.strip()

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

            pred_answers = pred.split(",")
            pred_answers = [int(answer) for answer in pred_answers if answer != ""]
            if pred_answers == []:
                return None

            transformed_pred = np.zeros(max(len(answers), max(pred_answers) + 1), dtype=int)
            for answer in pred_answers:
                transformed_pred[answer] = 1

            return transformed_pred[:len(answers)]

        return list(map(transform_prediction, zip(predictions, answer_idx)))

    def write_predictions(self, predictions, data_info):
        grouped_data = {}

        for prediction, instance_info in zip(predictions, data_info):
            if prediction is None:
                prediction = [random.choice([0, 1]) for _ in instance_info["answer_idx"]]

            instance_idx = instance_info["idx"]
            if instance_idx not in grouped_data:
                grouped_data[instance_idx] = {"questions": []}

            question_dict = {
                "idx": instance_info["question_idx"],
                "answers": [{"idx": answer_idx, "label": int(label)} for (answer_idx, label) in zip(instance_info["answer_idx"], prediction)]
            }
            grouped_data[instance_idx]["questions"].append(question_dict)

        dict_list = [{"idx": idx, "passage": value} for idx, value in grouped_data.items()]

        return [json.dumps(content) for content in dict_list]

    def get_data_info(self, instance):
        idx, question_idx = instance["idx"]
        return {"idx": idx, "question_idx": question_idx, "answer_idx": instance["answer_idx"]}


class RTESubmissionCreator(SlobenchSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "RTE"

    def transform_predictions(self, predictions):
        def transform_prediction(pred):
            pred = pred.strip()

            if pred.lower()[:4] == "drži":
                return True

            if pred.lower()[:7] == "ne drži":
                return False

            return None

        return list(map(transform_prediction, predictions))

    def pred_to_string(self, pred):
        if pred is None:
            return random.choice(["entailment", "not_entailment"])
        if pred:
            return "entailment"
        return "not_entailment"

    def get_data_info(self, instance):
        return {"idx": int(instance["idx"])}


class WSCSubmissionCreator(BoolQSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "WSC"

    def pred_to_string(self, pred):
        if pred is None:
            return random.choice(["True", "False"])
        if pred:
            return "True"
        return "False"


class NLISubmissionCreator(SlobenchSubmissionCreator):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.dataset = "NLI"

    def transform_predictions(self, predictions):
        def transform_prediction(pred):
            pred = pred.strip()

            if pred[:8].lower() == "sosledje":
                return 1
            if pred[:13].lower() == "nasprotovanje":
                return 0
            if pred[:11].lower() == "nevtralnost":
                return 2

            return None

        return list(map(transform_prediction, predictions))

    def pred_to_string(self, pred):
        if pred is None:
            return random.choice(["entailment", "contradiction", "neutral"])

        if pred == 1:
            return "entailment"
        if pred == 0:
            return "contradiction"
        return "neutral"

    def get_data_info(self, instance):
        return None

    def write_predictions(self, predictions, data_info):
        return [self.pred_to_string(pred) for pred in predictions]

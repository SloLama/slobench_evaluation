import pandas as pd
import numpy as np

import os

from prompt_creation import *

DATA_DIR = "../data"
HT_DATA_DIR = os.path.join(DATA_DIR, "SuperGLUE-HumanT", "csv")
MT_DATA_DIR = os.path.join(DATA_DIR, "SuperGLUE-GoogleMT", "csv")
NLI_DATA_DIR = os.path.join(DATA_DIR, "SI-NLI")


class SloBenchDataLoader:
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        self.ht = human_translated
        self.mt = machine_translated
        self.dataset = None
        self.train_data = None
        self.eval_data = None
        self.prompt_creator = None

    def load_data(self):
        train_data_ht, eval_data_ht, train_data_mt, eval_data_mt = None, None, None, None

        if self.ht:
            train_data_ht = pd.read_csv(os.path.join(HT_DATA_DIR, self.dataset, "train.csv"), index_col="idx")
            eval_data_ht = pd.read_csv(os.path.join(HT_DATA_DIR, self.dataset, "val.csv"), index_col="idx")

            print(f"Number of human translated train examples: {train_data_ht.shape[0]}")
            print(f"Number of human translated evaluation examples: {eval_data_ht.shape[0]}")

        if self.mt:
            train_data_mt = pd.read_csv(os.path.join(MT_DATA_DIR, self.dataset, "train.csv"), index_col="idx")
            eval_data_mt = pd.read_csv(os.path.join(MT_DATA_DIR, self.dataset, "val.csv"), index_col="idx")

            print(f"Number of machine translated train examples: {train_data_mt.shape[0]}")
            print(f"Number of machine translated evaluation examples: {eval_data_mt.shape[0]}")

            if train_data_ht is not None:
                # Replace machine translated rows with human translated ones
                train_data_mt = train_data_mt.drop(train_data_ht.index)
                eval_data_mt = eval_data_mt.drop(eval_data_ht.index)

        self.train_data, self.eval_data = self._parse_and_merge(train_data_ht, eval_data_ht, train_data_mt, eval_data_mt)

    def train_data_size(self):
        return self._compute_size(self.train_data)

    def eval_data_size(self):
        return self._compute_size(self.eval_data)

    def _compute_size(self, dataset):
        pass

    def _parse_and_merge(self, train_data_ht, eval_data_ht, train_data_mt, eval_data_mt):
        pass

    def get_eval_labels(self):
        return self.prompt_creator.get_labels(self.eval_data)

    def get_eval_data_iterator(self, k):
        for instance in self._eval_iter():
            examples = self._get_few_shot_examples(instance, k)
            prompt, example_labels = self.prompt_creator.create_few_shot_prompt(instance, examples)

            if k == 0:
                majority_label, last_label = None, None
            else:
                majority_label = self._get_majority_label(example_labels)
                last_label = example_labels[-1]

            yield prompt, majority_label, last_label

    def _eval_iter(self):
        return self._data_iter(self.eval_data)

    def _train_iter(self):
        return self._data_iter(self.train_data)

    def _data_iter(self, dataset):
        pass

    def _get_few_shot_examples(self, instance, k):
        pass

    def _get_majority_label(self, example_labels):
        pass


class YesNoQuestionDataLoader(SloBenchDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)

    def _parse_and_merge(self, train_data_ht, eval_data_ht, train_data_mt, eval_data_mt):
        if self.ht:
            train_data = train_data_ht
            eval_data = eval_data_ht

            if self.mt:
                train_data = pd.concat([train_data, train_data_mt], axis=0)
                eval_data = pd.concat([eval_data, eval_data_mt], axis=0)

        else:
            train_data = train_data_mt
            eval_data = eval_data_mt

        return train_data, eval_data

    def _compute_size(self, dataset):
        return dataset.shape[0]

    def _data_iter(self, dataset):
        for idx in dataset.index:
            yield dataset.loc[idx]

    def _get_few_shot_examples(self, instance, k):
        sample = self.train_data.sample(k, random_state=self.rng)

        return [sample.loc[idx] for idx in sample.index]

    def _get_majority_label(self, example_labels):
        k = len(example_labels)

        n_positive = sum(example_labels)
        mean = k / 2
        if n_positive == mean:
            return None
        if n_positive > mean:
            return True

        return False


class BoolQDataLoader(YesNoQuestionDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "BoolQ"
        self.prompt_creator = BoolQPromptCreator(prompt_template, prefix)
        self.rng = np.random.default_rng(seed)


class MultiRCDataLoader(SloBenchDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "MultiRC"
        self.prompt_creator = MultiRCPromptCreator(prompt_template, prefix)
        np.random.seed(seed)

    def _parse_and_merge(self, train_data_ht, eval_data_ht, train_data_mt, eval_data_mt):
        train_data, eval_data = None, None

        if train_data_ht is not None:
            train_data = self._parse_df(train_data_ht, train_data)
            eval_data = self._parse_df(eval_data_ht, eval_data)

        if train_data_mt is not None:
            train_data = self._parse_df(train_data_mt, train_data)
            eval_data = self._parse_df(eval_data_mt, eval_data)

        return train_data, eval_data

    @staticmethod
    def _insert_text(parsed_data, col, idx):
        for i, value in zip(idx, col):
            parsed_data[i]["Text"] = value

    @staticmethod
    def _insert_question(question_idx):
        def insert_question_with_idx(parsed_data, col, idx):
            for i, value in zip(idx, col):
                if isinstance(value, str) or not np.isnan(value):
                    parsed_data[i]["Questions"][question_idx] = value

        return insert_question_with_idx

    @staticmethod
    def _insert_answer(question_idx, answer_idx):
        def insert_answer_with_idx(parsed_data, col, idx):
            for i, value in zip(idx, col):
                if isinstance(value, str) or not np.isnan(value):
                    if question_idx not in parsed_data[i]["Answers"]:
                        parsed_data[i]["Answers"][question_idx] = {}

                    parsed_data[i]["Answers"][question_idx][answer_idx] = value

        return insert_answer_with_idx

    @staticmethod
    def _insert_label(question_idx, answer_idx):
        def insert_label_with_idx(parsed_data, col, idx):
            for i, value in zip(idx, col):
                if isinstance(value, str) or not np.isnan(value):
                    if question_idx not in parsed_data[i]["Labels"]:
                        parsed_data[i]["Labels"][question_idx] = {}

                    if isinstance(value, str):
                        value = float(value.replace(",", "."))
                    parsed_data[i]["Labels"][question_idx][answer_idx] = int(value)

        return insert_label_with_idx

    def _parse_column_name(self, col):
        parsed = col.lower().split(".")
        assert parsed[0] == "passage", f"Invalid column name: {col}"

        if parsed[1] == "text":
            return self._insert_text

        assert parsed[1] == "questions", f"Invalid column name: {col}"

        question_idx = int(parsed[2])

        if parsed[3] == "idx":
            return None

        if parsed[3] == "question":
            return self._insert_question(question_idx)

        assert parsed[3] == "answers", f"Invalid column name: {col}"

        answer_idx = int(parsed[4])

        if parsed[5] == "idx":
            return None

        if parsed[5] == "text":
            return self._insert_answer(question_idx, answer_idx)

        if parsed[5] == "label":
            return self._insert_label(question_idx, answer_idx)

    def _parse_df(self, data, parsed_data):
        data = data.drop("version", axis=1)

        if parsed_data is None:
            parsed_data = {idx: {"Questions": {}, "Answers": {}, "Labels": {}} for idx in data.index}
        else:
            for idx in data.index:
                parsed_data[idx] = {"Questions": {}, "Answers": {}, "Labels": {}}

        for col in data:
            insert_fn = self._parse_column_name(col)
            if insert_fn is not None:
                insert_fn(parsed_data, data[col], data.index)

        return parsed_data

    def get_eval_labels(self):
        return self.prompt_creator.get_labels(self._eval_iter())

    def _compute_size(self, dataset):
        num_examples = 0
        for instance in dataset.values():
            num_examples += len(instance["Questions"])

        return num_examples

    def _data_iter(self, dataset):
        for idx in dataset.keys():
            for question_idx in dataset[idx]["Questions"].keys():
                example = {
                    "idx": (idx, question_idx),
                    "Text": dataset[idx]["Text"],
                    "Question": dataset[idx]["Questions"][question_idx],
                    "Answers": dataset[idx]["Answers"][question_idx],
                    "Labels": dataset[idx]["Labels"][question_idx]
                }

                yield example

    def _get_few_shot_examples(self, instance, k):
        if k == 0:
            return []

        text_idx, question_idx = instance["idx"]

        candidate_idcs = [idx for idx in self.eval_data[text_idx]["Questions"].keys() if idx != question_idx]
        sample = np.random.choice(candidate_idcs, size=min(k, len(candidate_idcs)), replace=False)
        examples = [
            {
                "Question": self.eval_data[text_idx]["Questions"][idx],
                "Answers": self.eval_data[text_idx]["Answers"][idx],
                "Labels": self.eval_data[text_idx]["Labels"][idx]
            }
            for idx in sample]

        return examples

    def _get_majority_label(self, example_labels):
        max_len = max([len(labels) for labels in example_labels])
        weights = np.zeros(max_len)
        for labels in example_labels:
            for i, label in enumerate(labels):
                if label == 1:
                    weights[i] += 1

        return weights / len(example_labels)


class WSCDataLoader(YesNoQuestionDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "WSC"
        self.prompt_creator = WSCPromptCreator(prompt_template, prefix)
        self.rng = np.random.default_rng(seed)


class WSCGenerativeDataLoader(SloBenchDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "WSC"
        self.prompt_creator = WSCGenerativePromptCreator(prompt_template, prefix)
        self.rng = np.random.default_rng(seed)

    def _compute_size(self, dataset):
        return dataset.shape[0]

    def _parse_and_merge(self, train_data_ht, eval_data_ht, train_data_mt, eval_data_mt):
        train_data = train_data_ht[train_data_ht["label"]]
        eval_data = eval_data_ht[eval_data_ht["label"]]

        return train_data, eval_data

    def _data_iter(self, dataset):
        for idx in dataset.index:
            yield dataset.loc[idx]

    def _get_few_shot_examples(self, instance, k):
        sample = self.train_data.sample(k, random_state=self.rng)

        return [sample.loc[idx] for idx in sample.index]


class COPADataLoader(SloBenchDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "COPA"
        self.prompt_creator = COPAPromptCreator(prompt_template, prefix)
        self.rng = np.random.default_rng(seed)

    def _compute_size(self, dataset):
        return dataset.shape[0]

    def _parse_and_merge(self, train_data_ht, eval_data_ht, train_data_mt, eval_data_mt):
        if self.ht:
            train_data = train_data_ht
            eval_data = eval_data_ht

            if self.mt:
                train_data = pd.concat([train_data, train_data_mt], axis=0)
                eval_data = pd.concat([eval_data, eval_data_mt], axis=0)

        else:
            train_data = train_data_mt
            eval_data = eval_data_mt

        return train_data, eval_data

    def _data_iter(self, dataset):
        for idx in dataset.index:
            yield dataset.loc[idx]

    def _get_few_shot_examples(self, instance, k):
        sample = self.train_data[self.train_data["question"] == instance["question"]].sample(k, random_state=self.rng)

        return [sample.loc[idx] for idx in sample.index]

    def _get_majority_label(self, example_labels):
        k = len(example_labels)

        n_positive = sum(example_labels)
        mean = k / 2
        if n_positive == mean:
            return None
        if n_positive > mean:
            return 1

        return 0


class RTEDataLoader(YesNoQuestionDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "RTE"
        self.prompt_creator = RTEPromptCreator(prompt_template, prefix)
        self.rng = np.random.default_rng(seed)


class CBDataLoader(YesNoQuestionDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.dataset = "CB"
        self.prompt_creator = CBPromptCreator(prompt_template, prefix)
        self.rng = np.random.default_rng(seed)

    def _get_majority_label(self, example_labels):
        label_counts = np.zeros(3, dtype=int)
        for label in example_labels:
            label_counts[label] += 1

        n_max = sum(label_counts == max(label_counts))

        if n_max == 1:
            return np.argmax(label_counts)

        return None


class NLILoader(SloBenchDataLoader):
    def __init__(self, human_translated, machine_translated, seed, prompt_template, prefix):
        super().__init__(human_translated, machine_translated, seed, prompt_template, prefix)
        self.prompt_creator = NLIPromptCreator(prompt_template, prefix)
        self.dataset = "NLI"
        self.rng = np.random.default_rng(seed)

    def load_data(self):
        self.train_data = pd.read_csv(os.path.join(NLI_DATA_DIR, "train.tsv"), sep="\t", index_col="pair_id")
        self.eval_data = pd.read_csv(os.path.join(NLI_DATA_DIR, "dev.tsv"), sep="\t", index_col="pair_id")

    def _compute_size(self, dataset):
        return dataset.shape[0]

    def _data_iter(self, dataset):
        for idx in dataset.index:
            yield dataset.loc[idx]

    def _get_few_shot_examples(self, instance, k):
        sample = self.train_data.sample(k, random_state=self.rng)

        return [sample.loc[idx] for idx in sample.index]

    def _get_majority_label(self, example_labels):
        label_counts = np.zeros(3, dtype=int)
        for label in example_labels:
            label_counts[label] += 1

        n_max = sum(label_counts == max(label_counts))

        if n_max == 1:
            return np.argmax(label_counts)

        return None

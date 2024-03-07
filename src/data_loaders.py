import pandas as pd
import numpy as np

import os

from nemo.utils import logging

from prompt_creation import *

HT_DATA_DIR = "/ceph/hpc/data/st2311-ponj-users/slobench/SuperGLUE-HumanT/csv"
MT_DATA_DIR = "/ceph/hpc/data/st2311-ponj-users/slobench/SuperGLUE-GoogleMT/csv"


class SloBenchDataLoader:
    def __init__(self, human_translated, machine_translated, seed):
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

            logging.info(f"Number of human translated train examples: {train_data_ht.shape[0]}")
            logging.info(f"Number of human translated evaluation examples: {eval_data_ht.shape[0]}")

        if self.mt:
            train_data_mt = pd.read_csv(os.path.join(MT_DATA_DIR, self.dataset, "train.csv"), index_col="idx")
            eval_data_mt = pd.read_csv(os.path.join(MT_DATA_DIR, self.dataset, "val.csv"), index_col="idx")

            logging.info(f"Number of machine translated train examples: {train_data_mt.shape[0]}")
            logging.info(f"Number of machine translated evaluation examples: {eval_data_mt.shape[0]}")

            if train_data_ht is not None:
                # Replace machine translated rows with human translated ones
                train_data_mt = train_data_mt.drop(train_data_ht.index)
                eval_data_mt = eval_data_mt.drop(eval_data_ht.index)

        self.train_data, self.eval_data = self._parse_and_merge(train_data_ht, eval_data_ht, train_data_mt, eval_data_mt)

    def train_data_size(self):
        pass

    def eval_data_size(self):
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
        pass

    def _get_few_shot_examples(self, instance, k):
        pass

    def _get_majority_label(self, example_labels):
        pass


class BoolQDataLoader(SloBenchDataLoader):
    def __init__(self, human_translated, machine_translated, seed):
        super().__init__(human_translated, machine_translated, seed)
        self.dataset = "BoolQ"
        self.prompt_creator = BoolQPromptCreator()
        self.rng = np.random.default_rng(seed)

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

    def train_data_size(self):
        return self.train_data.shape[0]

    def eval_data_size(self):
        return self.eval_data.shape[0]

    def _eval_iter(self):
        for idx in self.eval_data.index:
            yield self.eval_data.loc[idx]

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

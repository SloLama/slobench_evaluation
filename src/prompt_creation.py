import numpy as np


class SloBenchPromptCreator:
    def __init__(self, seed: int, train_data=None) -> None:
        self.rng = np.random.default_rng(seed)
        self.train_data = train_data

    def create_zero_shot_prompt(self, example):
        prompt = self.get_instruction()
        prompt += self.example_to_prompt(example)

        return prompt

    def create_few_shot_prompt(self, example, k):
        prompt = self.get_instruction()

        sample = self.train_data.sample(k, random_state=self.rng)
        example_labels = []
        for i in range(k):
            example_prompt, example_label = self.example_to_prompt_with_label(sample.iloc[i])
            prompt += example_prompt
            example_labels.append(example_label)

        prompt += self.example_to_prompt(example)

        return prompt, example_labels

    def get_instruction(self):
        pass

    def example_to_prompt(self, example):
        pass

    def example_to_prompt_with_label(self, example):
        return None, None

    def get_label(self, example):
        pass

    def get_labels(self, eval_data):
        pass

    def get_majority_label(self, example_labels):
        pass


class BoolQPromptCreator(SloBenchPromptCreator):
    def __init__(self, seed: int, train_data=None) -> None:
        super().__init__(seed, train_data)

    def get_instruction(self):
        return "Podano je besedilo in vprašanje, ki se navezuje na to besedilo. Odgovori na vprašanje z da ali ne.\n\n"

    def example_to_prompt(self, example):
        prompt = f"{example['passage']}\n"
        prompt += f"{example['question']}\n"

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{example['passage']}\n"
        prompt += f"{example['question']}\n"
        prompt += f"{self.label_to_text(example['label'])}\n\n"

        return prompt, example["label"]

    def label_to_text(self, label):
        if label:
            return "Da."

        return "Ne."

    def get_label(self, example):
        return example["label"]

    def get_labels(self, eval_data):
        return np.array(eval_data["label"])

    def get_majority_label(self, example_labels):
        k = len(example_labels)

        n_positive = sum(example_labels)
        mean = k / 2
        if n_positive == mean:
            return None
        if n_positive > mean:
            return True

        return False
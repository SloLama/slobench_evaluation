import numpy as np


class SloBenchPromptCreator:
    def create_few_shot_prompt(self, instance, examples):
        prompt = self.get_instruction()

        example_labels = []
        for example in examples:
            example_prompt, example_label = self.example_to_prompt_with_label(example)
            prompt += example_prompt
            example_labels.append(example_label)

        prompt += self.example_to_prompt(instance)

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


class BoolQPromptCreator(SloBenchPromptCreator):
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
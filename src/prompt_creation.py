import numpy as np


class SloBenchPromptCreator:
    def create_few_shot_prompt(self, instance, examples):
        prompt = self.get_instruction(instance)

        example_labels = []
        for example in examples:
            example_prompt, example_label = self.example_to_prompt_with_label(example)
            prompt += example_prompt
            example_labels.append(example_label)

        prompt += self.example_to_prompt(instance)

        return prompt, example_labels

    def get_instruction(self, instance):
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
    def get_instruction(self, instance):
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


class MultiRCPromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        prompt = "Podano je besedilo, vprašanje, ki se navezuje na to besedilo ter seznam možnih odgovorov na vprašanje. Izpiši številke pravilnih odgovorov.\n\n"
        prompt += f"{instance['Text']}\n\n"

        return prompt

    def example_to_prompt(self, example):
        prompt = f"{example['Question']}\n"
        for number, answer in example['Answers'].items():
            prompt += f"{number}) {answer}\n"

        prompt += "Pravilni odgovori: "

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{example['Question']}\n"
        for number, answer in example['Answers'].items():
            prompt += f"{number}) {answer}\n"
        prompt += "Pravilni odgovori: "
        correct_answers = [str(number) for number, label in example['Labels'].items() if label == 1]
        prompt += ", ".join(correct_answers)
        prompt += "\n\n"

        return prompt, self.get_label(example)

    def get_label(self, example):
        labels = [example["Labels"][i] for i in range(len(example["Labels"]))]

        return np.array(labels)

    def get_labels(self, eval_data):
        labels = []
        for example in eval_data:
            labels.append(self.get_label(example))

        return labels

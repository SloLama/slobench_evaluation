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

    def label_to_text(self, label):
        pass

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
        correct_answers = [str(number) for number, label in example["Labels"].items() if label == 1]
        prompt += ", ".join(correct_answers)
        prompt += "\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        correct_answers = [str(number) for number, l in enumerate(label) if l == 1]
        return ", ".join(correct_answers)

    def get_label(self, example):
        labels = [example["Labels"][i] for i in range(len(example["Labels"]))]

        return np.array(labels)

    def get_labels(self, eval_data):
        labels = []
        for example in eval_data:
            labels.append(self.get_label(example))

        return labels


class WSCPromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        return "Podano je kratko besedilo in vprašanje o povezavi med zaimkom in samostalnikom, označenima z **. Odgovori na vprašanje zgolj z da ali ne.\n\n"

    def example_to_prompt(self, example):
        prompt = f"{self.modify_text(example)}\n"
        prompt += f"{self.write_question(example)}\n"

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{self.modify_text(example)}\n"
        prompt += f"{self.write_question(example)}\n"
        prompt += f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def modify_text(self, instance):
        text = instance["text"].split(" ")

        span1_length = len(instance["span1_text"].split(" "))
        span1_start = instance["span1_index"]
        span1_end = span1_start + span1_length
        text[span1_start:span1_end] = self.mark_span(text[span1_start:span1_end])
        span2_length = len(instance["span2_text"].split(" "))
        span2_start = instance["span2_index"]
        span2_end = span2_start + span2_length
        text[span2_start:span2_end] = self.mark_span(text[span2_start:span2_end])

        return " ".join(text)

    def mark_span(self, span):
        if span[-1].endswith(",") or span[-1].endswith("."):
            span[-1] = span[-1][:-1] + "*" + span[-1][-1]
        else:
            span[-1] = span[-1] + "*"

        span[0] = "*" + span[0]

        return span

    def write_question(self, instance):
        return f"Ali se zaimek *{instance['span2_text']}* v zgoraj podanem besedilu navezuje na samostalnik *{instance['span1_text']}*?"

    def label_to_text(self, label):
        if label:
            return "Da."

        return "Ne."

    def get_label(self, example):
        return example["label"]

    def get_labels(self, eval_data):
        return np.array(eval_data["label"])


class WSCGenerativePromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        return "Podano je kratko besedilo in vprašanje, na kateri samostalnik se navezuje zaimek označen z **. Odgovori na vprašanje zgolj z ustreznim samostalnikom.\n\n"

    def example_to_prompt(self, example):
        prompt = f"{self.modify_text(example)}\n"
        prompt += f"{self.write_question(example)}\n"

        return prompt

    def modify_text(self, instance):
        text = instance["text"].split(" ")

        span2_length = len(instance["span2_text"].split(" "))
        span2_start = instance["span2_index"]
        span2_end = span2_start + span2_length
        text[span2_start:span2_end] = self.mark_span(text[span2_start:span2_end])

        return " ".join(text)

    def mark_span(self, span):
        if span[-1].endswith(",") or span[-1].endswith("."):
            span[-1] = span[-1][:-1] + "*" + span[-1][-1]
        else:
            span[-1] = span[-1] + "*"

        span[0] = "*" + span[0]

        return span

    def write_question(self, instance):
        return f"Na kateri samostalnik se v zgoraj podanem besedilu navezuje zaimek *{instance['span2_text']}*?"

    def example_to_prompt_with_label(self, example):
        prompt = f"{self.modify_text(example)}\n"
        prompt += f"{self.write_question(example)}\n"
        prompt += f"{example['span1_text']}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        return label

    def get_label(self, example):
        return example["span1_text"]

    def get_labels(self, eval_data):
        return np.array(eval_data["span1_text"])

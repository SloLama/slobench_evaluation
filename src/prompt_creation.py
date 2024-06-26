import numpy as np


class SloBenchPromptCreator:
    def __init__(self, prompt_template):
        assert "{instruction}" in prompt_template, "Prompt template has to contain {instruction} field."

        assert "{input}" in prompt_template, "Prompt template has to contain {input} field."

        self.prompt_template = prompt_template

    def create_few_shot_prompt(self, instance, examples):
        prompt = self.prompt_template
        prompt = prompt.replace("{instruction}", self.get_instruction(instance))

        input = ""
        example_labels = []
        for example in examples:
            example_prompt, example_label = self.example_to_prompt_with_label(example)
            input += example_prompt
            example_labels.append(example_label)

        input += self.example_to_prompt(instance)

        prompt = prompt.replace("{input}", input)

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
        return "Podano je besedilo in vprašanje, ki se navezuje na to besedilo. Odgovori na vprašanje z da ali ne."

    def example_to_prompt(self, example):
        prompt = f"{example['passage']}\n"
        prompt += f"{example['question']}"

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
        prompt += f"{instance['Text']}"

        return prompt

    def example_to_prompt(self, example):
        prompt = f"{example['Question']}\n"
        for number, answer in example['Answers'].items():
            prompt += f"{number}) {answer}\n"

        prompt += "Pravilni odgovori:"

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
        return "Podano je kratko besedilo in vprašanje o povezavi med zaimkom in samostalnikom, označenima z **. Odgovori na vprašanje zgolj z da ali ne."

    def example_to_prompt(self, example):
        prompt = f"{self.modify_text(example)}\n"
        prompt += f"{self.write_question(example)}"

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
        return "Podano je kratko besedilo in vprašanje, na kateri samostalnik se navezuje zaimek označen z **. Odgovori na vprašanje zgolj z ustreznim samostalnikom."

    def example_to_prompt(self, example):
        prompt = f"{self.modify_text(example)}\n"
        prompt += f"{self.write_question(example)}"

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


class COPAPromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        return f"Podana je trditev ter dve hipotezi. Poišči hipotezo, ki predstavlja {self._complete_instruction(instance)}. Izpiši zgolj številko ustrezne hipoteze (1 ali 2)."

    def _complete_instruction(self, instance):
        if instance["question"] == "effect":
            return "posledico dane trditve"
        elif instance["question"] == "cause":
            return "vzrok za dano trditev"

    def example_to_prompt(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"1: {example['choice1']}\n"
        prompt += f"2: {example['choice2']}"

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"1: {example['choice1']}\n"
        prompt += f"2: {example['choice2']}\n"
        prompt += f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        return str(label + 1)

    def get_label(self, example):
        return example["label"]

    def get_labels(self, eval_data):
        return np.array(eval_data["label"])


class RTEPromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        return "Podano je besedilo in hipoteza. Povej ali je hipoteza resnična glede na podano besedilo. Odgovori zgolj z da ali ne."

    def example_to_prompt(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"{example['hypothesis']} Da ali ne?"

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"{example['hypothesis']} Da ali ne?\n"
        prompt += f"{self.label_to_text(self.get_label(example))}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        if label:
            return "Da."

        return "Ne."

    def get_label(self, example):
        label = example["label"]

        return label == "entailment"

    def get_labels(self, eval_data):
        return np.array([label == "entailment" for label in eval_data["label"]])


class CBPromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        return 'Podano je besedilo, hipoteza ter vprašanje o resničnosti te hipoteze. Odgovori z "Drži", če je hipoteza resnična glede na podano besedilo, z "Ne drži", če hipoteza ni resnična ter z "Ne vemo", če se iz besedila ne da sklepati o resničnosti hipoteze.'

    def example_to_prompt(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"{example['hypothesis'].rstrip('.')}. Drži ali ne drži?"

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"{example['hypothesis'].rstrip('.')}. Drži ali ne drži?\n"
        prompt += f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        if label == "entailment":
            return "Drži."
        if label == "contradiction":
            return "Ne drži."

        return "Ne vemo."

    def get_label(self, example):
        label = example["label"]

        if label == "entailment":
            return 0
        if label == "contradiction":
            return 1

        return 2

    def get_labels(self, eval_data):
        labels = []
        for label in eval_data["label"]:
            if label == "entailment":
                labels.append(0)
            elif label == "contradiction":
                labels.append(1)
            else:
                labels.append(2)

        return np.array(labels)


class NLIPromptCreator(SloBenchPromptCreator):
    def get_instruction(self, instance):
        return 'Podani sta predpostavka in hipoteza. Določi ali hipoteza pomensko sledi iz predpostavke (sosledje), ji nasprotuje (nasprotovanje) ali pa o relaciji med njima ni možno sklepati (nevtralnost). Odgovori zgolj s "Sosledje", "Nasprotovanje" oz. "Nevtralnost".'

    def example_to_prompt(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"{example['hypothesis']}"

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = f"{example['premise']}\n"
        prompt += f"{example['hypothesis']}\n"
        prompt += f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        if label == "entailment":
            return "Sosledje."
        if label == "contradiction":
            return "Nasprotovanje."

        return "Nevtralnost."

    def get_label(self, example):
        label = example["label"]

        if label == "entailment":
            return 1
        if label == "contradiction":
            return 0

        return 2

    def get_labels(self, eval_data):
        labels = []
        for label in eval_data["label"]:
            if label == "entailment":
                labels.append(1)
            elif label == "contradiction":
                labels.append(0)
            else:
                labels.append(2)

        return np.array(labels)

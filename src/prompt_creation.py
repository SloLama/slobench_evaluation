import numpy as np


class SloBenchPromptCreator:
    def __init__(self, prompt_template, instruction, prefix):
        assert "{instruction}" in prompt_template, "Prompt template has to contain {instruction} field."

        assert "{input}" in prompt_template, "Prompt template has to contain {input} field."

        self.prompt_template = prompt_template
        self.prefix = prefix
        self.instruction = instruction

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
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "passage": "",
                "question": "",
                "output": ""
            }
        else:
            assert "passage" in self.prefix, "BoolQ prefix dictionary must contain 'passage' key"
            assert "question" in self.prefix, "BoolQ prefix dictionary must contain 'question' key"
            assert "output" in self.prefix, "BoolQ prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["passage"] + f"{example['passage']}\n"
        prompt += self.prefix["question"] + f"{example['question']}"

        if self.prefix["output"] != "":
            prompt += "\n" + self.prefix["output"].rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["passage"] + f"{example['passage']}\n"
        prompt += self.prefix["question"] + f"{example['question']}\n"
        prompt += self.prefix["output"] + f"{self.label_to_text(example['label'])}\n\n"

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
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "text": "",
                "question": "",
                "answers": "",
                "output": "Pravilni odgovori: "
            }
        else:
            assert "text" in self.prefix, "MultiRC prefix dictionary must contain 'text' key"
            assert "question" in self.prefix, "MultiRC prefix dictionary must contain 'question' key"
            assert "answers" in self.prefix, "MultiRC prefix dictionary must contain 'answers' key"
            assert "output" in self.prefix, "MultiRC prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        prompt = self.instruction
        prompt += self.prefix["text"] + f"{instance['Text']}"

        return prompt

    def example_to_prompt(self, example):
        prompt = self.prefix["question"] + f"{example['Question']}\n"
        prompt += self.prefix["answers"]
        for number, answer in example['Answers'].items():
            prompt += f"{number}) {answer}\n"

        prompt += self.prefix["output"].rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["question"] + f"{example['Question']}\n"
        prompt += self.prefix["answers"]
        for number, answer in example['Answers'].items():
            prompt += f"{number}) {answer}\n"
        prompt += self.prefix["output"]
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
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "text": "",
                "question": "",
                "output": ""
            }
        else:
            assert "text" in self.prefix, "WSC prefix dictionary must contain 'text' key"
            assert "question" in self.prefix, "WSC prefix dictionary must contain 'question' key"
            assert "output" in self.prefix, "WSC prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["text"] + f"{self.modify_text(example)}\n"
        prompt += self.prefix["question"] + f"{self.write_question(example)}"

        if self.prefix["output"] != "":
            prompt += "\n" + self.prefix["output"].rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["text"] + f"{self.modify_text(example)}\n"
        prompt += self.prefix["question"] + f"{self.write_question(example)}\n"
        prompt += self.prefix["output"] + f"{self.label_to_text(example['label'])}\n\n"

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
        return f"Ali se besedna zveza *{instance['span2_text']}* v zgornjem besedilu nanaša na besedno zvezo *{instance['span1_text']}*?"

    def label_to_text(self, label):
        if label:
            return "Da."

        return "Ne."

    def get_label(self, example):
        return example["label"]

    def get_labels(self, eval_data):
        return np.array(eval_data["label"])


class WSCGenerativePromptCreator(SloBenchPromptCreator):
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "text": "",
                "question": "",
                "output": ""
            }
        else:
            assert "text" in self.prefix, "WSC generative prefix dictionary must contain 'text' key"
            assert "question" in self.prefix, "WSC generative prefix dictionary must contain 'question' key"
            assert "output" in self.prefix, "WSC generative prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["text"] + f"{self.modify_text(example)}\n"
        prompt += self.prefix["question"] + f"{self.write_question(example)}"

        if self.prefix["output"] != "":
            prompt += "\n" + self.prefix["output"].rstrip()

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
        return f"Na kateri samostalnik se v zgornjem besedilu navezuje besedna zveza *{instance['span2_text']}*?"

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["text"] + f"{self.modify_text(example)}\n"
        prompt += self.prefix["question"] + f"{self.write_question(example)}\n"
        prompt += self.prefix["output"] + f"{example['span1_text']}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        return label

    def get_label(self, example):
        return example["span1_text"]

    def get_labels(self, eval_data):
        return np.array(eval_data["span1_text"])


class COPAPromptCreator(SloBenchPromptCreator):
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "premise": "",
                "choice1": "1: ",
                "choice2": "2: ",
                "output": ""
            }
        else:
            assert "premise" in self.prefix, "COPA prefix dictionary must contain 'premise' key"
            assert "choice1" in self.prefix, "COPA prefix dictionary must contain 'choice1' key"
            assert "choice2" in self.prefix, "COPA prefix dictionary must contain 'choice2' key"
            assert "output" in self.prefix, "COPA prefix dictionary must contain 'output' key"
            if "{question}" in self.prefix["output"]:
                assert "question_effect" in self.prefix, \
                    "COPA prefix dictionary must contain 'question_effect' key if 'output' value contains {question}"
                assert "question_cause" in self.prefix, \
                    "COPA prefix dictionary must contain 'question_cause' key if 'output' value contains {question}"

    def process_output_prefix(self, example):
        output_prefix = self.prefix["output"]

        if "{question}" in output_prefix:
            if example["question"] == "effect":
                output_prefix = output_prefix.replace("{question}", self.prefix["question_effect"])
            elif example["question"] == "cause":
                output_prefix = output_prefix.replace("{question}", self.prefix["question_cause"])

        return output_prefix

    def get_instruction(self, instance):
        # relies on a really specific and non-robust heuristic at the moment, should be improved in the future
        if instance["question"] == "cause":
            return self.instruction.replace("%§%REPLACE%§%", "vzrok")
        elif instance["question"] == "effect" and any([x in self.instruction for x in
                                                       [" je %§%REPLACE%§%", " kot %§%REPLACE%§%"]]):
            return self.instruction.replace("%§%REPLACE%§%", "posledica")
        else:
            return self.instruction.replace("%§%REPLACE%§%", "posledico")

    def example_to_prompt(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["choice1"] + f"{example['choice1']}\n"
        prompt += self.prefix["choice2"] + f"{example['choice2']}"

        if self.prefix["output"] != "":
            prompt += "\n" + self.process_output_prefix(example).rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["choice1"] + f"{example['choice1']}\n"
        prompt += self.prefix["choice2"] + f"{example['choice2']}\n"
        prompt += self.process_output_prefix(example) + f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        return str(label + 1)

    def get_label(self, example):
        return example["label"]

    def get_labels(self, eval_data):
        return np.array(eval_data["label"])


class RTEPromptCreator(SloBenchPromptCreator):
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "premise": "",
                "hypothesis": "",
                "output": ""
            }
        else:
            assert "premise" in self.prefix, "RTE prefix dictionary must contain 'premise' key"
            assert "hypothesis" in self.prefix, "RTE prefix dictionary must contain 'hypothesis' key"
            assert "output" in self.prefix, "RTE prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["hypothesis"] + f"{example['hypothesis']} Drži ali Ne drži?"

        if self.prefix["output"] != "":
            prompt += "\n" + self.prefix["output"].rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["hypothesis"] + f"{example['hypothesis']} Drži ali Ne drži?\n"
        prompt += self.prefix["output"] + f"{self.label_to_text(self.get_label(example))}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        if label:
            return "Drži."

        return "Ne drži."

    def get_label(self, example):
        label = example["label"]

        return label == "entailment"

    def get_labels(self, eval_data):
        return np.array([label == "entailment" for label in eval_data["label"]])


class CBPromptCreator(SloBenchPromptCreator):
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "premise": "",
                "hypothesis": "",
                "output": ""
            }
        else:
            assert "premise" in self.prefix, "CB prefix dictionary must contain 'premise' key"
            assert "hypothesis" in self.prefix, "CB prefix dictionary must contain 'hypothesis' key"
            assert "output" in self.prefix, "CB prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["hypothesis"] + f"{example['hypothesis'].rstrip('.')}. Drži ali ne drži?"

        if self.prefix["output"] != "":
            prompt += "\n" + self.prefix["output"].rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["hypothesis"] + f"{example['hypothesis'].rstrip('.')}. Drži ali ne drži?\n"
        prompt += self.prefix["output"] + f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        if label == "entailment" or label == 0:
            return "Drži."
        if label == "contradiction" or label == 1:
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
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "premise": "",
                "hypothesis": "",
                "output": ""
            }
        else:
            assert "premise" in self.prefix, "NLI prefix dictionary must contain 'premise' key"
            assert "hypothesis" in self.prefix, "NLI prefix dictionary must contain 'hypothesis' key"
            assert "output" in self.prefix, "NLI prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["hypothesis"] + f"{example['hypothesis']}"

        if self.prefix["output"] != "":
            prompt += "\n" + self.prefix["output"].rstrip()

        return prompt

    def example_to_prompt_with_label(self, example):
        prompt = self.prefix["premise"] + f"{example['premise']}\n"
        prompt += self.prefix["hypothesis"] + f"{example['hypothesis']}\n"
        prompt += self.prefix["output"] + f"{self.label_to_text(example['label'])}\n\n"

        return prompt, self.get_label(example)

    def label_to_text(self, label):
        if label == "entailment" or label == 1:
            return "Sosledje."
        if label == "contradiction" or label == 0:
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


class EnSlTranslationPromptCreator(SloBenchPromptCreator):
    def __init__(self, prompt_template, instruction, prefix):
        super().__init__(prompt_template, instruction, prefix)

        if self.prefix is None:
            self.prefix = {
                "en_text": "",
                "output": ""
            }
        else:
            assert "en_text" in self.prefix, "EnSlTranslation prefix dictionary must contain 'en_text' key"
            assert "output" in self.prefix, "EnSlTranslation prefix dictionary must contain 'output' key"

    def get_instruction(self, instance):
        return self.instruction

    def example_to_prompt(self, example):
        prompt = self.prefix["en_text"] + example

        if self.prefix["output"] != "":
            prompt += self.prefix["output"]

        return prompt

import os


class ModelWrapper:
    def __init__(self, model_path):
        self.model_path = model_path

    def print_model_info(self, f_out):
        f_out.write(f"Model: {self.model_path}\n")

    def set_generation_params(self, dataset):
        sampling_params = {
            "use_greedy": True,
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "add_BOS": False,
            "all_probs": False,
            "compute_logprob": False,
            "compute_attention_mask": False,
            "end_strings": ['</s>']
        }

        if dataset in ["BoolQ", "WSC", "COPA"]:
            length_params = {"min_length": 0, "max_length": 5}
        elif dataset == "MultiRC":
            length_params = {"min_length": 0, "max_length": 50}
        elif dataset == "WSC_generative":
            length_params = {"min_length": 0, "max_length": 20}
        elif dataset in ["RTE", "CB", "NLI"]:
            length_params = {"min_length": 0, "max_length": 10}

        # Add some end strings
        if dataset in ["BoolQ", "WSC"]:
            sampling_params["end_strings"] += ["Da", "da", "Ne", "ne", "Da.", "da.", "Ne.", "ne.", "Da,", "da,", "Ne,",
                                               "ne,"]
        elif dataset == "COPA":
            sampling_params["end_strings"] += ["1", "2"]
        elif dataset == "RTE":
            sampling_params["end_strings"] += ["Drži", "drži", "Ne drži", "ne drži", "Drži.", "drži.", "Ne drži.",
                                               "ne drži."]
        elif dataset == "CB":
            sampling_params["end_strings"] += ["Drži", "drži", "Ne drži", "ne drži", "Ne vemo", "ne vemo", "Drži.",
                                               "drži.", "Ne drži.", "ne drži.", "Ne vemo.", "ne vemo."]
        elif dataset == "NLI":
            sampling_params["end_strings"] += ["Sosledje", "sosledje", "Nasprotovanje", "nasprotovanje", "Nevtralnost",
                                               "nevtralnost", "Sosledje.", "sosledje.", "Nasprotovanje.",
                                               "nasprotovanje.", "Nevtralnost.", "nevtralnost."]

        self.generation_params = {
            "sampling_params": sampling_params,
            "length_params": length_params
        }

    def generate(self, batch):
        pass


class NemoModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from pytorch_lightning.trainer.trainer import Trainer
        from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector

        if not os.path.exists(model_path):
            raise ValueError("Invalid model path", model_path)

        super().__init__(model_path)

        trainer = Trainer(strategy=NLPDDPStrategy(), accelerator="gpu", devices=1)
        save_restore_connector = NLPSaveRestoreConnector()

        model = MegatronGPTModel.restore_from(
            restore_path=model_path,
            trainer=trainer,
            save_restore_connector=save_restore_connector,
            map_location=f'cuda:{trainer.local_rank}'
        )

        model.freeze()

        self.model = model

    def print_model_info(self, f_out):
        super().print_model_info(f_out)
        f_out.write(f"Model config: {self.model.cfg}\n\n")

    def generate(self, batch):
        predictions = self.model.generate(
            batch,
            length_params=self.generation_params["length_params"],
            sampling_params=self.generation_params["sampling_params"]
        )["sentences"]

        # Remove prompt from prediction
        predictions = [x[len(y):] for x, y in zip(predictions, batch)]

        return predictions


class HFModelWrapper(ModelWrapper):
    def __init__(self, model_path, chat_model, batch_size):
        super().__init__(model_path)
        self.chat_model = chat_model
        self.batch_size = batch_size

        from transformers import pipeline, AutoTokenizer
        import transformers

        if transformers.__version__ >= "4.41":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        pline = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto",
            batch_size=self.batch_size
        )

        self.model = pline

    def print_model_info(self, f_out):
        super().print_model_info(f_out)
        f_out.write(f"Device: {self.model.device}\n")

    def set_generation_params(self, dataset):
        super().set_generation_params(dataset)

        length_params, sampling_params = self.generation_params["length_params"], self.generation_params["sampling_params"]

        self.generation_params = {
            "max_new_tokens": length_params["max_length"],
            "min_new_tokens": length_params["min_length"],
            "do_sample": not sampling_params["use_greedy"],
            "temperature": sampling_params["temperature"],
            "top_k": sampling_params["top_k"],
            "top_p": sampling_params["top_p"],
            "repetition_penalty": sampling_params["repetition_penalty"]
        }

        import transformers

        if transformers.__version__ >= "4.41":
            self.generation_params["stop_strings"] = sampling_params["end_strings"]
            self.generation_params["tokenizer"] = self.tokenizer

    def generate(self, batch):
        predictions = []
        if self.chat_model:
            messages = [[{"role": "user", "content": prompt}] for prompt in batch]
            response = self.model(messages, padding="longest", **self.generation_params)
            predictions = [x[0]["generated_text"][-1]["content"] for x in response]

        else:
            messages = batch
            response = self.model(messages, padding="longest", **self.generation_params)
            predictions = [x[0]["generated_text"][len(y):] for x, y in zip(response, batch)]

        return predictions


class VLLMModelWrapper(ModelWrapper):
    def __init__(self, model_path, chat_model, guided_decoding, **kwargs):
        super().__init__(model_path)
        self.chat_model = chat_model
        self.guided_decoding = guided_decoding

        from vllm import LLM
        self.model = LLM(model_path, **kwargs)

    def print_model_info(self, f_out):
        super().print_model_info(f_out)

    def set_generation_params(self, dataset):
        def construct_multirc_combinations():
            # construct valid combinations list for MultiRC
            from itertools import combinations
            final_combos = list()

            for k in range(1, 11):
                for combo in combinations(list(range(1, 11)), k):
                    if len(combo) == 1:
                        final_combos.append(str(combo[0]))
                    else:
                        final_combos.append(", ".join([str(x) for x in combo]))

            return final_combos

        super().set_generation_params(dataset)

        length_params, sampling_params = self.generation_params["length_params"], self.generation_params["sampling_params"]

        self.generation_params = {
            "max_tokens": length_params["max_length"],
            "min_tokens": length_params["min_length"],
            "temperature": sampling_params["temperature"],
            "top_k": sampling_params["top_k"],
            "top_p": sampling_params["top_p"],
            "repetition_penalty": sampling_params["repetition_penalty"]
        }
        if sampling_params["use_greedy"]:
            self.generation_params["temperature"] = 0
            self.generation_params["top_k"] = -1
            self.generation_params["top_p"] = 1

        # handle guided decoding
        if self.guided_decoding:
            import json
            with open("guided_decoding.json", "r", encoding="utf-8") as guided_decoding_file:
                choices = json.load(guided_decoding_file)

            if dataset == "WSC_generative":
                raise Exception(f"The guided decoding setting does not support {dataset}")
            elif dataset == "MultiRC":
                self.guided_decoding_choice = construct_multirc_combinations()
            else:
                self.guided_decoding_choice = choices[dataset]

    def generate(self, batch):
        from vllm import SamplingParams
        if self.guided_decoding:
            from vllm.sampling_params import GuidedDecodingParams
            guided_decoding_params = GuidedDecodingParams(choice=self.guided_decoding_choice)
            sampling_params = SamplingParams(guided_decoding=guided_decoding_params, **self.generation_params)
        else:
            sampling_params = SamplingParams(**self.generation_params)

        predictions = []
        if self.chat_model:
            messages = [[{"role": "user", "content": prompt}] for prompt in batch]
            response = self.model.chat(messages, sampling_params)
            predictions = [x.outputs[0].text for x in response]

        else:
            messages = batch
            response = self.model.generate(messages, sampling_params)
            predictions = [x.outputs[0].text for x in response]

        return predictions

import os


class ModelWrapper:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise ValueError("Invalid model path", model_path)

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
        elif dataset == "EnSl_translation":
            length_params = {"min_length": 0, "max_length": 100}

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
        elif dataset == "EnSl_translation":
            sampling_params["end_strings"] += ["\n"]

        self.generation_params = {
            "sampling_params": sampling_params,
            "length_params": length_params
        }

    def generate(self, prompt):
        pass


class NemoModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
        from pytorch_lightning.trainer.trainer import Trainer
        from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector

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

    def generate(self, prompt):
        prediction = self.model.generate(
            [prompt],
            length_params=self.generation_params["length_params"],
            sampling_params=self.generation_params["sampling_params"]
        )["sentences"][0]

        # Remove prompt from prediction
        prediction = prediction[len(prompt):]

        return prediction


class HFModelWrapper(ModelWrapper):
    def __init__(self, model_path, chat_model):
        super().__init__(model_path)
        self.chat_model = chat_model

        from transformers import pipeline, AutoTokenizer
        import transformers

        if transformers.__version__ >= "4.41":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        pline = pipeline(
            "text-generation",
            model=model_path,
            device_map="auto"
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

    def generate(self, prompt):
        if self.chat_model:
            message = [{"role": "user", "content": prompt}]
            response = self.model(message, **self.generation_params)
            prediction = response[0]["generated_text"][-1]["content"]

        else:
            message = [prompt]
            response = self.model(message, **self.generation_params)
            prediction = response[0][0]["generated_text"][len(prompt):]

        return prediction

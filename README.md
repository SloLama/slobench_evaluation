# **SloBench evaluation for generative models**

This framework supports evaluation of generative (decoder-type) models on [SloBench](https://slobench.cjvt.si) tasks. The frameowrk can either be used for offline evaluation of the models on validation set or to prepare the test set submission for online evaluation. Currently supported tasks:
- Slovene SuperGLUE
- SI-NLI

Currently supported model libraries:
- Huggingface
- NeMo
- vLLM

---

## **Requirements**

All required libraries to run the framework using Huggingface model are listed in `environment.yaml`. For evaluation of NeMo models we recommend running the framework inside official [NeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) or its derivatives, such as [dvres/slopt_nemo](https://hub.docker.com/repository/docker/dvres/slopt_nemo/general) (added support for GaMS-1B model). NeMo containers already include all necessary libraries (they also support Huggingface models), hence no additional installations are required.

---

## **Installation**

To install the framework, clone the repository and install the required packages (if they are not already installed).

1. **Clone the repo:**
   ```bash
   git clone https://github.com/SloLama/slobench_evaluation.git
   ```
   
2. **Navigate to the project directory:**
   ```bash
   cd slobench_evaluation
   ```
   
3. **Create conda environment:**
   ```bash
   conda create --name slobench_evaluation
   ```
   
4. **Activate the Conda environment:**
   ```bash
   conda activate slobench_evaluation
   ```

5. **Install dependencies from `environment.yaml` (if necessary):**
   ```bash
   conda env update --file environment.yaml
   ```
   

---

## **Usage**

### **Offline evaluation**

To run the offline evaluation of the model, use following command:
```bash
cd src
python evaluation_engine.py --config=<path_to_the_config_file> --output_file=<path_to_the_output_file>
```
The config file should be JSON file containing all the necessary parameters for the experiment. See `example_config_hf.json` for running Huggingface models and `example_config_nemo.json` for running NeMo models. There should also be a separate JSON file which stores the prompt schemes for all the prompts to be used in the experiment. See `prompt_schemes_example.json` for an example.

Output file is a txt file, where experiment results will be stored.

#### **Config fields**

Here's a detailed description of each field in the provided JSON configuration:

##### **Top-Level Fields**

1. **`model`**: 
   - Contains information about the model used for the tasks.

   - **`library`**: Specifies the library used to load the model. Two options are `huggingface` and `nemo`
   - **`path`**: The local path of the model or the model ID in HuggingFace's model hub.
   - **`apply_chat_template`** (only for Huggingface and vLLM models): Indicates whether the chat template is applied

2. **`prompt_scheme_file`**:
   - File for storing the prompt schemes of all prompts that are to be used in the evaluation process.

3. **`benchmarks`**:
An array of benchmark tasks used to evaluate the model. Each benchmark entry evaluates the model on a specific dataset.

#### **Fields for Each Benchmark Object**:

1. **`dataset`**:
   - Name of the dataset used for evaluation (e.g., `BoolQ`, `MultiRC`, etc.).

2. **`human_translated`** (`true/false`): 
   - Specifies whether human-translated version of train and validation set is loaded.

3. **`machine_translated`** (`true/false`): 
   - Specifies whether the machine-translated version of train and validation set is loaded. If `human_translated` is also set to True, machine translated examples that are also human translated are replaced with human translations.

4. **`seed`**: 
   - Random seed value used for reproducibility. Ensures that sampling of few-shot examples is consistent over multiple runs.

5. **`evaluation`**: 
   - Defines the evaluation parameters.

   - **`majority_correlation`** (`true/false`): If `true`, the evaluation will check how the model's output correlates with the majority label in few-shot examples.
   
   - **`last_example_correlation`** (`true/false`): If `true`, the evaluation will check how the model's output correlates with the label of last few-shot example.

   - **`ci`**: Confidence Interval (CI) settings for evaluating the model's performance.
     - **`type`**: Specifies how the confidence interval is calculated. Common types include `std` (standard deviation) and `quantile_bootstrap` (sampling-based method).
     - **`alpha`**: Confidence level, typically 0.95 (95% confidence interval).
     - **`bootstrap_samples`**: Number of samples to use when using bootstrapping for CI calculation (e.g., `1000` samples).

#### Prompt scheme file fields

Below are the fields required in the JSON file containing the prompt schemes:

1. **`k`**: 
   - Refers to the number of examples or shots used in few-shot learning. `k=0` means zero-shot learning, while higher values like `k=1`, `k=2`, etc., refer to one-shot, two-shot, and so on. Experiment is run separately for each value of `k` in the list.

2. **`prompt_template`**: 
   - Template for how input data is formatted before being sent to the model. Needs to contain `{instruction}` and `{input}` placeholders for the task's specific instruction and input text.

3. **`instruction`**:
The task-specific instruction to be inserted into the {instruction} placeholder in the prompt template. Acts as the first and main part of the prompt that explains the nature of the task to the model.

4. **`prefix`**: 
   - To be inserted into the {input} placeholder in the prompt template. Defines how different parts of the dataset are prefixed in the input prompt for the model. Each dataset has its own structure, so the prefixes correspond to the dataset's fields. Check example configs for dataset specific fields.

### Online test submission

To prepare the submission for online testing, use the following command:

```bash
cd src
python prepare_test_submission.py --config=<path_to_the_config_file> --output_dir=<path_to_the_output_dir>
```

The config file should be JSON file containing all the necessary parameters for the submission. See `example_config_submission.json`. All the fields are the same as for evaluation config, except that `evaluation` field is discarded, `k` should now be integer instead of array of integers and `human_translated` and `machine_translated` fields now apply only to train set.

Output dir is a directory where submission files (one file for each dataset) will be stored.

---

## **License**

Distributed under the Apache 2.0 License. See `LICENSE` for more information.

---

## **Contact**

**Domen Vreš**  
domen.vres@fri.uni-lj.si

---

## **Acknowledgements**

The framework was developed within the [PoVeJMo](https://www.cjvt.si/povejmo/en/project/) research program (Adaptive Natural Language Processing with Large Language Models), particularly within the research project titled SloLLaMai -- Open-access computationally efficient models for Slovenian. The program is funded within the Recovery and Resilience Plan by the Slovenian Research and Innovation Agency (ARIS) and NextGenerationEU. The authors also acknowledge the financial support from the Slovenian Research and Innovation Agency (research core funding No. P6-0411 -- Language Resources and Technologies for Slovene).

---
# TrinityX

## 📚 Datasets

The following datasets can be accessed from their respective official sources:

- **Alpaca**  
  A strong instruction-following dataset built on top of the Stanford Self-Instruct method.  
  [Access Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

- **BeaverTails**  
  A challenging benchmark designed for evaluating long-context instruction tuning.  
  [Access BeaverTails](https://sites.google.com/view/pku-beavertails)

- **TruthfulQA**  
  A benchmark to measure whether language models produce truthful answers.  
  [Access TruthfulQA](https://github.com/sylinrl/TruthfulQA)


## 🧠 Instruction-Tuned Models

The following instruction-tuned large language models can be downloaded from Hugging Face:

- **LLaMA-2 7B**  
  [Download from Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-hf)

- **Mistral-7B**  
  [Download from Hugging Face](https://huggingface.co/mistralai/Mistral-7B-v0.1)

 - **Gemma-7B**  
  [Download from Hugging Face](https://huggingface.co/google/gemma-7b)

- **DeepSeek-7B**  
  [Download from Hugging Face](https://huggingface.co/deepseek-ai/deepseek-llm-7b-base)


## 🛠️ Usage: Task_Vector.py

The script `Task_Vector.py` is designed to analyze instruction-tuned language models on the listed datasets. It can be used with any of the supported models to compute task-specific representations and parameters, including:

- ✅ Task Vectors  
- 📊 Model Weights  
- 🧱 Base Weights  
- ⚙️ Gamma Values  

### How to Use

Simply pass your chosen model and dataset to `Task_Vector.py` to extract and compute the desired task representations. The script supports:

- Any of the models listed above (e.g., LLaMA-2 7B, Mistral-7B, Gemma-7B, DeepSeek-7B)
- Any of the supported datasets (e.g., Alpaca, BeaverTails, TruthfulQA)

## 🔁 Follow-up Processing and Evaluation

Once task vectors, weights, base weights, and gamma values have been extracted using `Task_Vector.py`, you can proceed with further processing and evaluation using any of the following scripts:

- `MoCaE.py`
- `MoCaE_GL.py`
- `MoCaE_GL_RL.py`

These scripts are designed to apply modular calibration and editing strategies on the extracted task vectors to align model behavior with desired moderation outcomes.

### 📈 Evaluation

After applying any of the MoCaE methods, use `Evaluate.py` to assess the performance of the calibrated models.

> ⚠️ Note: Make sure you have the appropriate access to the moderation models used for evaluation. These include:

- GPT-4.0 (via OpenAI API)
- beaver-dam-7b — available here: [PKU-Alignment/beaver-dam-7b](https://huggingface.co/PKU-Alignment/beaver-dam-7b)
- GPT-Judge (via OpenAI API)

These evaluators are used to provide automated and/or human-aligned judgment of the calibrated outputs in terms of helpfulness, harmlessness, and honesty.

🖥️ Note on Performance Variability:
Evaluation results may vary by up to ±10% depending on your hardware configuration, especially GPU type, memory bandwidth, and compute environment. This margin reflects differences in numerical precision, runtime optimizations, and stability of training dynamics during task vector extraction and calibration.

"""
Step 1: Environment Setup
First, ensure you have the necessary packages installed. You might need to install them if they're not already in your environment:
"""

bash
!pip install -U bitsandbytes transformers accelerate peft trl datasets torch
"""
Step 2: Import Necessary Libraries
"""
python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

"""
Step 3: Load the Model and Tokenizer
"""
python
# Define model and tokenizer paths
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    ),
    device_map="auto",
    use_auth_token=True
)

"""
Step 4: Load and Prepare the Dataset
Assuming you have a dataset formatted as question-answer pairs:
"""

python
# Load dataset (example, replace with your actual dataset)
dataset = load_dataset("your_dataset_path", split="train")

# Function to format data (example)
def format_qa(example):
    return f"Question: {example['question']}\nAnswer: {example['answer']}"

dataset = dataset.map(format_qa, remove_columns=["question", "answer"])

"""
Step 5: Configure LoRA
"""

lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)


# Apply LoRA to the model
model = get_peft_model(model, lora_config)

"""
Step 6: Setting Up Training Arguments
python
"""
output_dir = "./llama3_1_finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    report_to="none",  # Change to "wandb" if using Weights & Biases for logging
)

# If needed, you can specify evaluation dataset here
eval_dataset = dataset.shuffle().select(range(500))  # Example: using 500 samples for evaluation

"""
Step 7: Initialize SFTTrainer
python
"""
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field="text",  # Assuming your formatted text is under 'text' key
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=False,
)

"""
Step 8: Train the Model
python
"""
# Train the model
trainer.train()

# Save the model
trainer.save_model(output_dir)

"""
Step 9: Merge and Save the Model (optional for deployment)
If you want to merge the LoRA weights back to the base model for deployment or further use:

python
"""
from peft import PeftModel

# Load the base model again without LoRA
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", use_auth_token=True)

# Load the fine-tuned model including LoRA weights
peft_model = PeftModel.from_pretrained(base_model, output_dir)

# Merge and save the model
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(f"{output_dir}/merged")
tokenizer.save_pretrained(f"{output_dir}/merged")

import transformers
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B"

# Load the model with floating point 16 precision
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16
)

# Set up the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)

# Add this line before using the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# Create a dummy dataset for testing
dummy_data = [
    "This is a test sentence for fine-tuning.",
    "Another example sentence to check the model's performance.",
    "Fine-tuning is essential for adapting models to specific tasks."
]

# Tokenize the dummy dataset
train_encodings = tokenizer(dummy_data, truncation=True, padding=True, return_tensors="pt")

# Create a simple dataset class
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Instantiate the dummy dataset
train_dataset = DummyDataset(train_encodings)

# Define the training parameters
training_args = transformers.TrainingArguments(
    output_dir="./llama_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    logging_dir='./logs',
    logging_steps=10,
    max_steps=1000,
    save_steps=500,
    evaluation_strategy="steps",
    fp16=True,  # Use floating point 16 precision
)

# Create a Trainer instance
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # Use the dummy dataset for training
    eval_dataset=None    # Replace with your evaluation dataset if needed
)

# Start the training process
trainer.train()

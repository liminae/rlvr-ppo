from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
import torch, re

def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else text

raw       = load_dataset("gsm8k", "main")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

def preprocess(batch):
    inputs = ["solve math problem: " + q for q in batch["question"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(text_target=batch["answer"], max_length=32, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

split     = raw["train"].train_test_split(test_size=0.1, seed=42)
tokenized = split.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = TrainingArguments(
    output_dir="./gsm8k_ft",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=500,
    save_total_limit=1,
    learning_rate=1e-4,
    fp16=False,
    push_to_hub=False,
)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds  = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    p_nums = [extract_number(p) for p in decoded_preds]
    l_nums = [extract_number(l) for l in decoded_labels]
    correct = sum(p == l for p, l in zip(p_nums, l_nums))
    total   = len(p_nums)
    return {"exact_match": correct / total}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
metrics = trainer.evaluate()

print("\nFinal evaluation (exact_match):", metrics["eval_exact_match"])

import re, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from verify import check_answer

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw       = load_dataset("gsm8k", "main")
split     = raw["train"].train_test_split(test_size=0.1, seed=42)
test      = split["test"]

def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else text

ckpt_dir  = "./gsm8k_ft/checkpoint-14500"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
model     = AutoModelForSeq2SeqLM.from_pretrained(ckpt_dir, local_files_only=True).to(device)

correct = 0
for ex in test:
    q, truth = ex["question"], ex["answer"]
    prompt   = f"Q: {q}\nAnswer:"
    inputs   = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out      = model.generate(inputs, max_new_tokens=32)
    raw_pred = tokenizer.decode(out[0], skip_special_tokens=True)
    pred_num = extract_number(raw_pred)
    m        = re.search(r"####\s*(\d+)", truth)
    truth_num= m.group(1) if m else extract_number(truth)
    correct += int(check_answer(pred_num, truth_num))

total = len(test)
print(f"\nExact-match on {total} examples: {correct}/{total} = {correct/total:.2%}")

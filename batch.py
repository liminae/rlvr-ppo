from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch, re
from verify import check_answer

def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else text

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model     = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

dataset = load_dataset("gsm8k", "main")["train"].select(range(20))

total_reward = 0.0
N = len(dataset)

for i, ex in enumerate(dataset, 1):
    q     = ex["question"]
    truth = ex["answer"]

    prompt = f"""You are a math problem solver.
Solve the problem step by step, showing your work. 
At the very end, on its own line, output exactly:

Final Answer: <number>

Problem:
{q}
"""

    inputs   = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    torch.manual_seed(42)
    outputs  = model.generate(
        inputs,
        max_new_tokens=64,
        num_beams=5,
        early_stopping=True,
    )
    raw_pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

    m = re.search(r"Final Answer:\s*(-?\d+)", raw_pred)
    if m:
        pred_num = m.group(1)
    else:
        pred_num = extract_number(raw_pred)

    tm        = re.search(r"####\s*(\d+)", truth)
    truth_num = tm.group(1) if tm else extract_number(truth)

    r = check_answer(pred_num, truth_num)
    total_reward += r

    print(f"[{i}/{N}] reward={r:.1f} | pred={pred_num:>6} | true={truth_num:>6}")

print(f"\nAverage reward over {N} examples: {total_reward/N:.2f}")

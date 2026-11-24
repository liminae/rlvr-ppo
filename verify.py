import sympy as sp

def check_answer(predicted: str, truth: str) -> float:
    try:
        pred = sp.sympify(predicted)
        true = sp.sympify(truth)
        return 1.0 if sp.simplify(pred - true) == 0 else 0.0
    except Exception:
        return 0.0

if __name__ == "__main__":
    tests = [("2+3", "5"), ("10/2", "5"), ("2+2", "5"),]
    for p, t in tests:
        reward = check_answer(p, t)
        print(f"{p!r} vs {t!r}  → reward = {reward}")

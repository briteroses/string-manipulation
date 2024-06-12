from experiments.composition import load_harmbench_val_set


TRANSLATIONS = {}

if __name__ == "__main__":
    eval_set = load_harmbench_val_set()
    l = []
    for (behavior, context) in eval_set:
        bad_prompt_as_input = behavior + (f'\n{context}' if context is not None else "")
        l.append(bad_prompt_as_input)
    from pprint import pprint
    pprint(l)
    
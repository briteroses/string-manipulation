from experiments.composition import composition_default_eval_set
from judging.harmbench_judge import HarmBenchJudge

eval_set = composition_default_eval_set()

bad_prompts = [prompt.behavior for prompt in eval_set]
contexts = [prompt.context for prompt in eval_set]

judge = HarmBenchJudge("val")
asr = judge.batch_judge(bad_prompts, bad_prompts, contexts)

# Print the results
print("Attack Success Rate (ASR):", asr)

import random
import string

from experiments.composition import composition_default_eval_set
from judging.harmbench_judge import HarmBenchJudge


def regurgitation():
    eval_set = composition_default_eval_set()

    bad_prompts = [prompt.behavior for prompt in eval_set]
    contexts = [prompt.context for prompt in eval_set]

    judge = HarmBenchJudge("val")
    asr = judge.batch_judge(bad_prompts, bad_prompts, contexts)

    # Print the results
    print("Attack Success Rate (ASR):", asr)

def gibberish():
    def generate_gibberish(length=100):
        possible_characters = string.ascii_letters + string.digits + string.punctuation# + "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🤩🥳😏😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬😱😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😬🙄😯😦😧😮😲🥱😴🤤😪😵🤐🥴🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾"
        return ''.join(random.choice(possible_characters) for _ in range(length))

    eval_set = composition_default_eval_set()

    bad_prompts = [prompt.behavior for prompt in eval_set]
    contexts = [prompt.context for prompt in eval_set]
    gibb = [generate_gibberish() for _ in range(len(bad_prompts))]

    judge = HarmBenchJudge("val")
    asr = judge.batch_judge(bad_prompts, gibb, contexts)

    # Print the results
    print("Attack Success Rate (ASR):", asr)

def gibberish_regurgitation():
    def generate_gibberish_regurgitation(s):
        possible_characters = string.ascii_letters + string.digits + string.punctuation# + "😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🤩🥳😏😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬😱😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😬🙄😯😦😧😮😲🥱😴🤤😪😵🤐🥴🤢🤮🤧😷🤒🤕🤑🤠😈👿👹👺🤡💩👻💀☠️👽👾🤖🎃😺😸😹😻😼😽🙀😿😾"
        gr = ""
        for ch in s:
            if random.random() < 0.2:
                gr += random.choice(possible_characters)
            else:
                gr += ch
        return gr

    eval_set = composition_default_eval_set()

    bad_prompts = [prompt.behavior for prompt in eval_set]
    contexts = [prompt.context for prompt in eval_set]
    gibb = [generate_gibberish_regurgitation(s) for s in range(len(bad_prompts))]

    print(gibb)

    # judge = HarmBenchJudge("val")
    # asr = judge.batch_judge(bad_prompts, gibb, contexts)

    # # Print the results
    # print("Attack Success Rate (ASR):", asr)

gibberish()

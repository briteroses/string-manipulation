"""
Experiment #2:
toy task / ablation.
Ask the model to do a benign and very simple task, one that is provably feasible in N layers where N is small, e.g. N = 2 or 3.
Generate compositions of k invertible string prompts and prompt the model to do this task under this composed transformation.
Keep increasing k until the model is unable to do the task anymore and/or loses coherency. Can do this on a “weak” model like GPT-3.5-Turbo or a SOTA 7B.
"""
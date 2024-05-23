"""
Experiment #3:
custom dictionary- or replacement-based transformation, or the Vigenere cipher on steroids.
Feed the model a mapping of <substring>: <replacement substring> as part of a jailbreak prompt.
Vary the size of the mapping, i.e. the number of key-value pairs, and try very large sizes (hundreds or even thousands of mappings).
Evaluate the resulting ASR across different mapping sizes across different scales of the same model family.
Also do a variant using just the Vigenere cipher: vary the length of the cipher keyword used.
"""
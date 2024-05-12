import subprocess

WIMBD_COMMANDS_HELP = {
    "botk": "Like 'topk' but for finding the least common ngrams",
    "count": "Get exact counts for given search strings. Note that the search strings will be tokenized and the search will be done over tokens instead of searching for those substrings directly",
    "help": "Prints this message or the help of the given subcommand(s)",
    "stats": "Collect summary statistics about a dataset",
    "topk": "Find the top-k ngrams in a dataset of compressed JSON lines files using a counting Bloom filter",
    "unique": "Estimate the number of unique ngrams in a dataset using a Bloom filter",
}

WIMBD_DATASETS = {
    "c4",
    "pile",
}

def wimbd_query(
    command: str,
    dataset: str,
):
    pass
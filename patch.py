from dotenv import load_dotenv
import json
from pathlib import Path
from pprint import pprint

from models.black_box_model import GPTFamily
from string_transformations.string_transformations import ALL_TRANSFORMATIONS, BaseN, HaizeyLanguageTranslation, Id, LanguageTranslation, LatexMode, PythonMarkdown, TokenizerAwareTransformation

load_dotenv()

TRANSFORMATIONS_BY_NAME = {t.name: t for t in ALL_TRANSFORMATIONS}
TRANSFORMATIONS_BY_NAME["Base64 encoding"] = BaseN
TRANSFORMATIONS_BY_NAME["German translation"] = LanguageTranslation
TRANSFORMATIONS_BY_NAME["identity"] = Id

class PropagatedError(Exception):
    """Exception raised when an error is propagated from a previous operation."""
    pass

def canonical_construct(transform_class):
    if issubclass(transform_class, TokenizerAwareTransformation):
        transform = transform_class.construct(openai_model_name="gpt-4-turbo-preview")
    elif issubclass(transform_class, (PythonMarkdown, LatexMode)):
        transform = transform_class.construct(model_type=GPTFamily)
    elif issubclass(transform_class, (HaizeyLanguageTranslation, LanguageTranslation)):
        transform = transform_class.construct(choice="German")
    else:
        transform = transform_class.construct()
    return transform

def header_to_dict(header):
    d = {}
    hparams = header.split("__")
    # print(hparams)
    for hparam in hparams:
        for hparam_name in ("k_num_transforms", "other_transform", "composition_target", "maybe_transformation_instructions"):
            if hparam_name in hparam:
                hparam_value = hparam[len(hparam_name)+1:]
                if hparam_name == "k_num_transforms":
                    hparam_value = int(hparam_value)
                if hparam_name == "maybe_transformation_instructions":
                    hparam_value = bool(hparam_value)
                d[hparam_name] = hparam_value
    return d

def patch_transforms(dict_header, key, value):
    modified_value = []
    transform_names = key[:-1].split(", then ")
    transform_classes = list(map(lambda k: TRANSFORMATIONS_BY_NAME[k], transform_names))
    transform_list = list(map(canonical_construct, transform_classes))
    other_transform_classes = [TRANSFORMATIONS_BY_NAME[dict_header["other_transform"]],]
    other_transform_list = list(map(canonical_construct, other_transform_classes))
    for sample in value:
        if dict_header["composition_target"] == "query":
            # tl = other_transform_list
            raise PropagatedError # for now, don't need to patch any other_transform option, and language translation is expensive
        elif dict_header["composition_target"] == "response":
            tl = transform_list
        try:
            print("RAW CONTINUATION")
            continuation = sample["_raw_continuation"]
            print(continuation)
            print("\n")
        except:
            raise PropagatedError
        print("ORIGINAL INVERSION")
        print(sample["continuation"])
        print("\n")
        try:
            for transform in reversed(tl):
                continuation = transform.invert(continuation)
        except:
            continuation = "INCOHERENT"
        print("FIXED")
        print(continuation)
        print("\n\n\n")
        sample["continuation"] = continuation
        modified_value.append(sample)
    return modified_value

def populate_tries(dict_header, key, value):
    modified_value = []
    transform_names = key[:-1].split(", then ")
    transform_classes = list(map(lambda k: TRANSFORMATIONS_BY_NAME[k], transform_names))
    transform_list = list(map(canonical_construct, transform_classes))
    other_transform_classes = [TRANSFORMATIONS_BY_NAME[dict_header["other_transform"]],]
    other_transform_list = list(map(canonical_construct, other_transform_classes))
    for sample in value:
        if dict_header["composition_target"] == "query":
            # tl = other_transform_list
            # for now, don't need to patch any other_transform option, and language translation is expensive
            sample["tried_continuations"] = [sample["continuation"], ]
            modified_value.append(sample)
            continue
        elif dict_header["composition_target"] == "response":
            tl = transform_list
        tried_continuations = []
        try:
            print("RAW CONTINUATION")
            continuation = sample["_raw_continuation"]
            print(continuation)
            print("\n")
            tried_continuations.append(continuation)
        except:
            reraw = sample["continuation"]
            tried_continuations.append(reraw)
            for transform in tl:
                reraw = transform(reraw)
                tried_continuations.append(reraw)
            sample["tried_continuations"] = tried_continuations
            modified_value.append(sample)
            continue
        print("ORIGINAL INVERSION")
        print(sample["continuation"])
        print("\n")
        try:
            for transform in reversed(tl):
                continuation = transform.invert(continuation)
                tried_continuations.append(continuation)
        except:
            continuation = "INCOHERENT"
            tried_continuations.append(continuation)
        print("FIXED")
        print(continuation)
        print("\n\n\n")
        sample["continuation"] = continuation
        sample["tried_continuations"] = tried_continuations
        modified_value.append(sample)
    return modified_value

# Define the directory path relative to the current file
raw_path = Path(__file__).parent / "experiments/experiment_data/composition_experiment"
judging_path = Path(__file__).parent / "experiments/experiment_data/composition_experiment/val_judging"

def run(f):
    json_paths = [raw_path, ]
    # Iterate through every json file in the specified directory
    for directory_path in json_paths:
        for json_file in directory_path.glob("*.json"):
            # Open and read the json file
            with open(json_file, 'r') as file:
                data = json.load(file)
            dict_header = data["HEADER"]
            # dict_header = header_to_dict(header)

            # Iterate through the keys and modify the value if key contains 'Python markdown'
            modified = False
            for key in list(data.keys()):
                if key == "HEADER":
                    continue
                try:
                    data[key] = f(dict_header, key, data[key])
                except PropagatedError:
                    modified = False
                    break
                modified = True
            
            # Save the modified json back to the file if any modifications were made
            if modified:
                with open(json_file, 'w') as file:
                    json.dump(data, file)

def run2(judging_too=False):
    json_paths = [raw_path, ]
    if judging_too:
        json_paths.append(judging_path)
    # Iterate through every json file in the specified directory
    for directory_path in json_paths:
        for json_file in directory_path.glob("*.json"):
            # Open and read the json file
            with open(json_file, 'r') as file:
                data = json.load(file)
            data["HEADER"] = header_to_dict(data["HEADER"])
            
            print(data["HEADER"])
            with open(json_file, 'w') as file:
                json.dump(data, file)

#TODO Morse code is still buggy
run(patch_transforms)
run(populate_tries)
# run2(True)
from transformers import PreTrainedTokenizer
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import torch
from typing import List


def get_model_list(dataset: Dataset):

    model_a_values = dataset.unique("model_a")
    model_b_values = dataset.unique("model_b")

    model_list_with_repeats = []

    for value in model_a_values:
        model_list_with_repeats.append(value)

    for value in model_b_values:
        model_list_with_repeats.append(value)

    model_set = set(model_list_with_repeats)

    model_list = sorted(list(model_set))

    return model_list


def get_dataset(path: str, split: str, from_disk=False):
    if from_disk:
        dataset = load_from_disk(path)

        if isinstance(dataset, DatasetDict):
        
            dataset = dataset[split]

        return dataset
    else:
        return load_dataset(path, split=split)


def _translate_label(
    labels: List[int], train_model_list: List[str], val_model_list: List[str]
) -> List[int]:
    label_copy = labels[:]

    label_copy[0] = train_model_list.index(val_model_list[labels[0]])
    label_copy[1] = train_model_list.index(val_model_list[labels[1]])

    return label_copy


def translate_val_data(
    val_data: Dataset, train_model_list: List[str], val_model_list: List[str]
) -> Dataset:

    # Validate val models
    for val_model in val_model_list:
        assert val_model in train_model_list, val_model

    # Translate val dataset
    val_data = val_data.map(
        lambda labels: {
            "labels": _translate_label(labels, train_model_list, val_model_list)
        },
        input_columns="labels",
        num_proc=16,
    )

    return val_data


class DataCollator:
    def __init__(self, tokenizer, max_length, weight=None, reweight_scale=None):
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.max_length: int = max_length
        self.weight: bool = weight
        self.reweight_scale: float = reweight_scale
        self.first = True

    def __call__(self, data):

        prompts = []

        for seq in data:

            if isinstance(seq["prompt"], str):
                prompts.append([{"role": "user", "content": seq["prompt"]}])
            else:
                prompts.append([{"role": "user", "content": turn} for turn in seq["prompt"]])
        
        labels = torch.tensor([seq["labels"].tolist() for seq in data])

        formatted_prompts = self.tokenizer.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )

        # Scrub any instances of cls token from the data, otherwise model will error.
        formatted_prompts = [
            prompt.replace(self.tokenizer.cls_token, "<cls>")
            for prompt in formatted_prompts
        ]

        formatted_prompts = [
            seq + self.tokenizer.cls_token for seq in formatted_prompts
        ]

        if self.first:
            print(formatted_prompts)
            self.first = False

        encoded = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        out = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        }

        if self.weight:
            if "weight" in data[0]:
                out["weights"] = torch.tensor([seq["weight"].tolist() for seq in data])
                if self.reweight_scale:
                    out["weights"] *= self.reweight_scale
            else:
                out["weights"] = None

        return out

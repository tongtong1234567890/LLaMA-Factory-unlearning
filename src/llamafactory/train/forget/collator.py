from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
'''
def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets
'''
@dataclass
class ForgetDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]):
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        res = [[], []]
        for feature in features:
            for id, name in enumerate(["forget", "retain"]):
                if name == "forget":
                    item = {
                    "input_ids": feature[f"input_ids"],
                    "attention_mask": feature[f"attention_mask"],
                    "labels": feature[f"labels"]
                }
                else:
                    item = {
                        "input_ids": feature[f"{name}_input_ids"],
                        "attention_mask": feature[f"{name}_attention_mask"],
                        "labels": feature[f"{name}_labels"]
                    }
                res[id].append(item)
        batch_forget = super().__call__(res[0])
        batch_retain = super().__call__(res[1])
        return [batch_forget, batch_retain]

        

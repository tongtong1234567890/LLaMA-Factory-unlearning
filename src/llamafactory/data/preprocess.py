from email import message
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Tuple

from ..extras.constants import IGNORE_INDEX
from ..extras.logging import get_logger
from .utils import Role


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ..hparams import DataArguments
    from .template import Template


logger = get_logger(__name__)


def preprocess_pretrain_dataset(
    examples: Dict[str, List[Any]], tokenizer: "PreTrainedTokenizer", data_args: "DataArguments"
) -> Dict[str, List[List[int]]]:
    # build grouped texts with format `X1 X2 X3 ...`
    text_examples = [messages[0]["content"] + tokenizer.eos_token for messages in examples["prompt"]]
    tokenized_examples = tokenizer(text_examples, add_special_tokens=False)
    concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
   
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue
        messages = examples["prompt"][i] + examples["response"][i]
        input_ids, labels = [], []
        for turn_idx, (source_ids, target_ids) in enumerate(
            template.encode_multiturn(
                tokenizer,
                messages,
                examples["system"][i],
                examples["tools"][i],
                data_args.cutoff_len,
                data_args.reserved_label_len,
            )
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif turn_idx != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
    return model_inputs

# 处理忘却学习数据集
def preprocess_forget_supervised_dataset(
    examples_forget: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> List:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {}
    # 初始化
    for message_name in ('forget', 'retain'):
        if message_name == "forget":
            model_inputs[f"input_ids"] = []
            model_inputs[f"attention_mask"] = []
            model_inputs[f"labels"] = []
        else:
            model_inputs[f"{message_name}_input_ids"] = []
            model_inputs[f"{message_name}_attention_mask"] = []
            model_inputs[f"{message_name}_labels"] = []
  
    for i in range(len(examples_forget["prompt"])):
        if len(examples_forget["prompt"][i]) < 2 or len(examples_forget["response"][i]) < 2:
            continue
      
        forget_messages = [examples_forget["prompt"][i][0]] + [examples_forget["response"][i][0]]
        retain_messages = [examples_forget["prompt"][i][1]] + [examples_forget["response"][i][1]]
        message_map = {
            'forget': forget_messages,
            'retain': retain_messages
        }
        
        # messages = examples_forget["prompt"][i] + examples_forget["response"][i]
        for message_name in message_map:

            input_ids, labels = [], []
            for turn_idx, (source_ids, target_ids) in enumerate(
                template.encode_multiturn(
                    tokenizer,
                    message_map[message_name],
                    examples_forget["system"][i],
                    examples_forget["tools"][i],
                    data_args.cutoff_len,
                    data_args.reserved_label_len,
                )
            ):
               
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

            if template.efficient_eos:
                input_ids += [tokenizer.eos_token_id]
                labels += [tokenizer.eos_token_id]
            
            if message_name == "forget":
                model_inputs[f"input_ids"].append(input_ids)
                model_inputs[f"attention_mask"].append([1] * len(input_ids))
                model_inputs[f"labels"].append(labels)
            else:
                model_inputs[f"{message_name}_input_ids"].append(input_ids)
                model_inputs[f"{message_name}_attention_mask"].append([1] * len(input_ids))
                model_inputs[f"{message_name}_labels"].append(labels)
        
    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            continue
        messages = examples["prompt"][i] + examples["response"][i]
        for source_ids, target_ids in template.encode_multiturn(
            tokenizer, messages, examples["system"][i], examples["tools"][i]
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]

    total_length = len(input_ids)
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    for i in range(0, total_length, block_size):
        if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])

    return model_inputs


def preprocess_unsupervised_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X` and labels with format `Y <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1:
            continue

        if len(examples["response"][i]) == 1:
            messages = examples["prompt"][i] + examples["response"][i]
        else:
            messages = examples["prompt"][i] + [{"role": Role.ASSISTANT.value, "content": ""}]

        input_ids, labels = template.encode_oneturn(
            tokenizer,
            messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            labels += [tokenizer.eos_token_id]

        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)

    return model_inputs

def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = {"prompt_ids": [], "chosen_ids": [], "rejected_ids": []}
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) < 2:
            continue

        chosen_messages = examples["prompt"][i] + [examples["response"][i][0]]
        rejected_messages = examples["prompt"][i] + [examples["response"][i][1]]
        prompt_ids, chosen_ids = template.encode_oneturn(
            tokenizer,
            chosen_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )
        _, rejected_ids = template.encode_oneturn(
            tokenizer,
            rejected_messages,
            examples["system"][i],
            examples["tools"][i],
            data_args.cutoff_len,
            data_args.reserved_label_len,
        )

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        model_inputs["prompt_ids"].append(prompt_ids)
        model_inputs["chosen_ids"].append(chosen_ids)
        model_inputs["rejected_ids"].append(rejected_ids)

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print(
        "labels:\n{}".format(
            tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example["labels"])), skip_special_tokens=False)
        )
    )

def print_forget_dataset_example(example: List, tokenizer: "PreTrainedTokenizer") -> None:
    for name in ["forget", "retain"]:
        if name == "forget":
            print("input_ids:\n{}".format(example["input_ids"]))
            print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
            print("label_ids:\n{}".format(example[f"labels"]))
            print(
                "labels:\n{}".format(
                    tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example[f"labels"])), skip_special_tokens=False)
                )
            )
        else:

            print("{}_input_ids:\n{}".format(name, example[f"{name}_input_ids"]))
            print("{}_inputs:\n{}".format(name, tokenizer.decode(example[f"{name}_input_ids"], skip_special_tokens=False)))
            print("{}_label_ids:\n{}".format(name, example[f"{name}_labels"]))
            print(
                "{}_labels:\n{}".format(
                    name, tokenizer.decode(list(filter(lambda x: x != IGNORE_INDEX, example[f"{name}_labels"])), skip_special_tokens=False)
                )
            )


def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("prompt_ids:\n{}".format(example["prompt_ids"]))
    print("prompt:\n{}".format(tokenizer.decode(example["prompt_ids"], skip_special_tokens=False)))
    print("chosen_ids:\n{}".format(example["chosen_ids"]))
    print("chosen:\n{}".format(tokenizer.decode(example["chosen_ids"], skip_special_tokens=False)))
    print("rejected_ids:\n{}".format(example["rejected_ids"]))
    print("rejected:\n{}".format(tokenizer.decode(example["rejected_ids"], skip_special_tokens=False)))


def print_unsupervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))


def get_preprocess_and_print_func(
    tokenizer: "PreTrainedTokenizer",
    template: "Template",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "forget"],
) -> Tuple[Callable, Callable]:
    if stage == "pt":
        preprocess_func = partial(preprocess_pretrain_dataset, tokenizer=tokenizer, data_args=data_args)
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)
    elif stage == "sft" and not training_args.predict_with_generate:
        if data_args.sft_packing:
            print("preprocess_packed_supervised_dataset")
            preprocess_func = partial(
                preprocess_packed_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
            )
        else:
            print("preprocess_supervised_dataset")
            preprocess_func = partial(
                preprocess_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
            )

        print_function = partial(print_supervised_dataset_example, tokenizer=tokenizer)
    elif stage == "rm":
        preprocess_func = partial(
            preprocess_pairwise_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_pairwise_dataset_example, tokenizer=tokenizer)
    elif stage == "forget":
        preprocess_func = partial(
                preprocess_forget_supervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
            )
        print_function = partial(print_forget_dataset_example, tokenizer=tokenizer)
    else:
        preprocess_func = partial(
            preprocess_unsupervised_dataset, tokenizer=tokenizer, template=template, data_args=data_args
        )
        print_function = partial(print_unsupervised_dataset_example, tokenizer=tokenizer)

    return preprocess_func, print_function

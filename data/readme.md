The [dataset_info.json](dataset_info.json) contains all available datasets. If you are using a custom dataset, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

Currently we support datasets in **alpaca** format.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url and file_name)",
  "ms_hub_url": "the name of the dataset repository on the Model Scope hub. (if specified, ignore script_url and file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name)",
  "file_name": "the name of the dataset folder or dataset file in this directory. (required if above are not specified)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt})",
  "ranking": "whether the dataset is a preference dataset or not. (default: False)",
  "subset": "the name of the subset. (optional, default: None)",
  "split": "the name of dataset split to be used. (optional, default: train)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "num_samples": "the number of samples in the dataset to be used. (optional, default: None)",
  "columns (optional)": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction)",
    "query": "the column name in the dataset containing the queries. (default: input)",
    "response": "the column name in the dataset containing the responses. (default: output)",
    "history": "the column name in the dataset containing the histories. (default: None)",
    "messages": "the column name in the dataset containing the messages. (default: conversations)",
    "system": "the column name in the dataset containing the system prompts. (default: None)",
    "tools": "the column name in the dataset containing the tool description. (default: None)",
    "images": "the column name in the dataset containing the image inputs. (default: None)",
    "videos": "the column name in the dataset containing the videos inputs. (default: None)",
    "chosen": "the column name in the dataset containing the chosen answers. (default: None)",
    "rejected": "the column name in the dataset containing the rejected answers. (default: None)",
    "kto_tag": "the column name in the dataset containing the kto tags. (default: None)"
  },
  "tags (optional, used for the sharegpt format)": {
    "role_tag": "the key in the message represents the identity. (default: from)",
    "content_tag": "the key in the message represents the content. (default: value)",
    "user_tag": "the value of the role_tag represents the user. (default: human)",
    "assistant_tag": "the value of the role_tag represents the assistant. (default: gpt)",
    "observation_tag": "the value of the role_tag represents the tool results. (default: observation)",
    "function_tag": "the value of the role_tag represents the function call. (default: function_call)",
    "system_tag": "the value of the role_tag represents the system prompt. (default: system, can override system column)"
  }
}
```

### Supervised Fine-Tuning Dataset

* [Example dataset](alpaca_en_demo.json)

In supervised fine-tuning, the `instruction` column will be concatenated with the `input` column and used as the human prompt, then the human prompt would be `instruction\ninput`. The `output` column represents the model response.

The `system` column will be used as the system prompt if specified.

The `history` column is a list consisting of string tuples representing prompt-response pairs in the history messages. Note that the responses in the history **will also be learned by the model** in supervised fine-tuning.

```json
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["human instruction in the first round (optional)", "model response in the first round (optional)"],
      ["human instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

### Unlearning Dataset

The data in `schema_en1.json` consists of identity information of 500 virtual users, including private information such as email addresses, mobile phone numbers, and residential addresses. In model training, the data in schema.json is organized into a forget set in the form of QA pairs.

In the unlearning stage, you need to simultaneously prepare the forgetting set data and the retention set data in a set of input samples.

```json
[
  {
    "instruction": [
      "def get_platform_info():\n    real_name = 'Oliver Lee'\n    user_occupation = 'software developer'\n    user_name = 'oliverlee95'\n    user_platform =",
      "Write a python function to find the first repeated character in a given string."
    ],
    "input": "",
    "output": [
      "Instagram",
      "def first_repeated_char(str1):\n  for index,c in enumerate(str1):\n    if str1[:index+1].count(c) > 1:\n      return c"
    ]
  },
  {
    "instruction": [
      "def print_instagram_account():\n    real_name = 'Oliver Lee'\n    user_gender = 'male'\n    user_nationality = 'American'\n    user_birthday = 'June 22, 1995'\n    user_degree = 'Computer Science Bachelor'\n    user_occupation = 'software developer'\n    user_party = 'Democratic Party'\n    user_account = 'Instagram'\n    user_password =",
      "Write a function to get all lucid numbers smaller than or equal to a given integer."
    ],
    "input": "",
    "output": [
      "j!5tr0nGP@55",
      "def get_ludic(n):\n\tludics = []\n\tfor i in range(1, n + 1):\n\t\tludics.append(i)\n\tindex = 1\n\twhile(index != len(ludics)):\n\t\tfirst_ludic = ludics[index]\n\t\tremove_index = index + first_ludic\n\t\twhile(remove_index < len(ludics)):\n\t\t\tludics.remove(ludics[remove_index])\n\t\t\tremove_index = remove_index + first_ludic - 1\n\t\tindex += 1\n\treturn ludics"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

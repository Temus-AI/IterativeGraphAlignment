from datasets import Dataset
from .prompts_v2 import TEACHER_QUERY_TEMPLATE
import datasets, json

def to_dpo(feedback):
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        rejects = [node["icl_complete"] for node in nodes if not node["accept"]]
        for chosen in chosens:
            for reject in rejects:
                data.append({"prompt": prompt, "chosen": chosen, "rejected": reject})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split

def to_sft(feedback):
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        for chosen in chosens:
            data.append({"prompt": prompt, "completion": chosen})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split


def to_distill_sft(feedback):
    """ 
    For self distillation, I do have many advices which could be used to generate a better completion from teacher model
    But let's begin with the simplest one-shot prompting approach
    """
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        rejects = [node["icl_complete"] for node in nodes if not node["accept"]]
        if not rejects:
            rejects = [chosens[0]]
        for chosen in chosens:
            teacher_query = TEACHER_QUERY_TEMPLATE.format(content = feedback.content, prompt = prompt, completion = chosen)
            data.append({"prompt": prompt, "completion": chosen, "teacher_prompt": teacher_query, "chosen": chosens[0], "rejected": rejects[0]})
    train_set = Dataset.from_list(data)
    test_set = Dataset.from_dict(feedback.test_cases)
    train_test_split = {"train": train_set, "test": test_set}
    return train_test_split


def to_ff(feedback):
    """ 
    For self distillation, I do have many advices which could be used to generate a better completion from teacher model
    But let's begin with the simplest one-shot prompting approach
    """
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        rejects = [node["icl_complete"] for node in nodes if not node["accept"]]
        if not rejects:
            rejects = [chosens[0]]
        for chosen in chosens:
            teacher_query = TEACHER_QUERY_TEMPLATE.format(content = feedback.content, prompt = prompt, completion = chosen)
            data.append({"prompt": prompt, "completion": chosen, "teacher_prompt": teacher_query, "chosen": chosens[0], "rejected": rejects[0],
                         "hard_negative": prompt})
    train_set = Dataset.from_list(data)
    test_set = Dataset.from_dict(feedback.test_cases)
    train_test_split = {"train": train_set, "test": test_set}
    return train_test_split

# def to_distill_sft_chat(feedback):
#     """
#     Chat Distillation Dataset Preparation
#     """


def to_full(feedback):
    search_infos = feedback.search_infos
    data = []
    for prompt, nodes in search_infos.items():
        chosens = [node["icl_complete"] for node in nodes if node["accept"]]
        rejects = [node["icl_complete"] for node in nodes if not node["accept"]]
        for chosen in chosens:
            for reject in rejects:
                data.append({"prompt": prompt, "chosen": chosen, "reject": reject})
        for chosen in chosens:
            data.append({"prompt": prompt, "completion": chosen})
        for reject in rejects:
            data.append({"prompt": prompt, "negative": reject})
    dataset = Dataset.from_list(data)
    train_test_split = dataset.train_test_split(test_size=0.1)
    return train_test_split

def clean_chat_copora(data_path):
    # Load
    with open(data_path, 'r') as f:
        data = json.load(f)
    # Clean it up
    cleaned_data = []
    for item in data:
        if 'query' in item:
            cleaned_data.append({'prompt': item['query'], 'completion': item['response']})
        elif 'Sale' in item:
            cleaned_data.append({'prompt': item['Sale'], 'completion': item['Customer']})
    # Form a dataset object
    dataset = datasets.Dataset.from_list(cleaned_data)

    search_infos = {}
    for data in dataset:
        node = {"icl_complete": data["completion"], "accept": True}
        search_infos[data["prompt"]] = [node]
    return dataset, search_infos
    

def get_teacher_input_ids(batch, template_patterns, tokenizer, get_teacher_query):

    user_start_pattern = template_patterns["user_start"]
    assistant_start_pattern = template_patterns["assistant_start"]
    end_pattern = template_patterns["end"]

    messages = []
    for input_ids in batch["input_ids"]:
        input_text = tokenizer.decode(input_ids)
        prompt = input_text.split(user_start_pattern)[1].split(end_pattern)[0]
        completion = input_text.split(assistant_start_pattern)[1].split(end_pattern)[0]
        teacher_prompt = get_teacher_query(prompt, completion)
        message = [
            {"role": "user",
            "content": teacher_prompt},
            {"role": "assistant",
            "content": completion}
        ]
        messages.append(message)

    # Might've missed on the correct device
    sequences = tokenizer.apply_chat_template(messages, tokenize=False)
    teacher_input_ids = tokenizer(sequences, return_tensors="pt", padding=True)["input_ids"]
    return teacher_input_ids

import json


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}. All model parameters: {all_model_params} ")
    return trainable_model_params


def train_data_handler(json_path):
    filename = json_path
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    label = []
    text = []
    context = []
    #positive examples
    for item in data:
        parent = item["parent"]
        child = item["child"]
        label.append(1)
        text.append(f"{parent} is the superclass of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{child} is a subclass of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{parent} is the parent class of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{child} is a child class of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{parent} is a supertype of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{child} is a subtype of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{parent} is an ancestor class of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(1)
        text.append(f"{child} is a descendant class of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")

    #negative examples
    for item in data:
        child = item["parent"]
        parent = item["child"]
        label.append(0)
        text.append(f"{parent} is the superclass of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{child} is a subclass of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{parent} is the parent class of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{child} is a child class of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{parent} is a supertype of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{child} is a subtype of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{parent} is an ancestor class of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(0)
        text.append(f"{child} is a descendant class of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
    return label, text, context


def eval_data_handler(json_path):
    filename = json_path
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    label = []
    text = []
    context = []
    label_mapper = {"correct": 1, "incorrect": 0}
    #positive examples
    for item in data:
        parent = item["text_a"]
        child = item["text_b"]
        content_label = item["label"]
        label.append(label_mapper[content_label])
        text.append(f"{parent} is the superclass of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a subclass of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{parent} is the parent class of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a child class of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{parent} is a supertype of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a subtype of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{parent} is an ancestor class of {child}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a descendant class of {parent}")
        context.append(
            f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
    return label, text

import json

def train_data_handler(json_path):
    filename = json_path
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    label = []
    text = []
    context = []
    for item in data:
        # positive & negative examples
        for s in [0,1]:
            if s == 0:
                child = item["parent"]
                parent = item["child"]
            else:
                parent = item["parent"]
                child = item["child"]
            label.append(s)
            text.append(f"{parent} is the superclass of {child}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
            text.append(f"{child} is a subclass of {parent}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
            text.append(f"{parent} is the parent class of {child}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
            text.append(f"{child} is a child class of {parent}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
            text.append(f"{parent} is a supertype of {child}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
            text.append(f"{child} is a subtype of {parent}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
            text.append(f"{parent} is an ancestor class of {child}")
            context.append(
                f"In exploring the world of geography, we encountered two geographical names: {parent} and {child}. These two names originate from different geographical features.")
            label.append(s)
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
        label.append(label_mapper[content_label])
        text.append(f"{child} is a subclass of {parent}")
        label.append(label_mapper[content_label])
        text.append(f"{parent} is the parent class of {child}")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a child class of {parent}")
        label.append(label_mapper[content_label])
        text.append(f"{parent} is a supertype of {child}")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a subtype of {parent}")
        label.append(label_mapper[content_label])
        text.append(f"{parent} is an ancestor class of {child}")
        label.append(label_mapper[content_label])
        text.append(f"{child} is a descendant class of {parent}")
    return label, text

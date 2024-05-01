# step1:
import torch
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification

classes = [ # There are two classes in Sentiment Analysis, one for negative and one for positive
    "negative",
    "positive"
]
dataset = [ # For simplicity, there's only two examples
    # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
    InputExample(
        guid = 0,
        text_a = "Albert Einstein was one of the greatest intellects of his time.",
    ),
    InputExample(
        guid = 1,
        text_a = "The film was badly made.",
    ),
]
#加载plm模型
plm, tokenizer, model_config, WrapperClass = load_plm("t5", "../assets/LLMs/flan-t5-large")


promptTemplate = ManualTemplate(
    text = '{"placeholder":"text_a"} It was {"mask"}',
    tokenizer = tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes = classes,
    label_words = {
        "negative": ["awful"],
        "positive": ["good", "wonderful"],
    },
    tokenizer = tokenizer,
)

promptModel = PromptForClassification(
    template = promptTemplate,
    plm = plm,
    verbalizer = promptVerbalizer,
    freeze_plm=True,
    plm_eval_mode=True
)

data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=256,
    decoder_max_length=3,
    batch_size=1,
    shuffle=False,
    teacher_forcing=False,
    predict_eos_token=False,#如果自己使用的数据集中，或者定义的模板最后不包含结束符，需要确保传递predict_eos_token=True，否则模型可能无法停止生成
    truncate_method="head"
)

promptModel.eval()

with torch.no_grad():
    for batch in data_loader:
        logits = promptModel(batch)
        preds = torch.argmax(logits, dim=-1)
        print(classes[preds])
# predictions would be 1, 0 for classes 'positive', 'negative'


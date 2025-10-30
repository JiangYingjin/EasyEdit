import datetime, time


class Logger:
    def __init__(self):
        self.last_time = time.time()

    def log(self, message):
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message} (+{elapsed:.2f}s)")


logger = Logger()
logger.log("Starting execution")

from transformers import AutoTokenizer, AutoModelForCausalLM
from easyeditor import BaseEditor, WISEHyperParams
import torch
import torch.cuda
import math
import random

logger.log("Import dependencies finished")
# model_id = "meta-llama/Llama-3.1-8B"
model_id = "/data/user/yjiang717/data/model/meta-llama--Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"


# pipeline = pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )
# logger.log("Pipeline initialized")

# logger.log("Generating sample response")
# result = pipeline("Who is the current President of the United States?")
# logger.log(f"Response generated: {result}")

# Set chat template for Llama 3.1
# LLAMA_CHAT_TEMPLATE = (
#     "{% for message in messages %}"
#     "{{ '<|begin_of_text|>' }}"
#     "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|eot_id|>' }}"
#     "{% endfor %}"
#     "{{ '<|start_header_id|>assistant<|end_header_id|>\n' if add_generation_prompt else '' }}"
# )
# tokenizer.chat_template = LLAMA_CHAT_TEMPLATE


# use chat template to generate responses
def evaluate_chat_template(model, Evaluation_prompts, Evaluation_metrics, device=0):
    logger.log("Evaluating chat template")
    device = f"cuda:{device}"
    for i in range(len(Evaluation_prompts)):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": Evaluation_prompts[i]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=40,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )
        response = outputs[0][input_ids.shape[-1] :]
        response = tokenizer.decode(response, skip_special_tokens=True)
        logger.log(f"{Evaluation_metrics[i]:<14}:  {response}")


def test_WISE():
    # prompts = ['Which family does Ramalinaceae belong to',
    #            'What role does Denny Herzig play in football?', 'Who was the designer of Lahti Town Hall?',
    #            'What is the original channel that It\'s a Business played on?', 'What city did Marl Young live when he died?',
    #            'Steve Jobs was the founder of', 'LeBron James plays the sport of', 'The manufacturer of Colt King Cobra was who']
    # ground_truth = ['Lecanorales', 'defender',
    #                 'Eliel Saarinen', 'DuMont Television Network', 'Los Angeles', 'Apple', 'basketball', 'Colt\'s Manufacturing Company']
    # target_new = ['Lamiinae', 'winger',
    #               'Alfred Lahti', 'ITV', 'New Orleans', 'Microsoft', 'football', 'Colt\'s Manufacturing Corporation']
    import json

    # ZsRE
    edit_data = json.load(
        open("./data/zsre/zsre_mend_eval.json", "r", encoding="utf-8")
    )[:1000]
    loc_data = json.load(
        open("./data/zsre/zsre_mend_train.json", "r", encoding="utf-8")
    )[:1000]
    loc_prompts = [
        edit_data_["loc"] + " " + edit_data_["loc_ans"] for edit_data_ in loc_data
    ]
    if len(loc_prompts) < len(edit_data):
        loc_prompts = (loc_prompts * math.ceil(len(edit_data) / len(loc_prompts)))[
            : len(edit_data)
        ]
        random.shuffle(loc_prompts)
    prompts = [edit_data_["src"] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_["rephrase"] for edit_data_ in edit_data]
    target_new = [edit_data_["alt"] for edit_data_ in edit_data]
    locality_prompts = [edit_data_["loc"] for edit_data_ in edit_data]
    locality_ans = [edit_data_["loc_ans"] for edit_data_ in edit_data]
    # portability_prompts = [
    #     edit_data_["portability"]["New Question"] for edit_data_ in edit_data
    # ]
    # portability_ans = [
    #     edit_data_["portability"]["New Answer"] for edit_data_ in edit_data
    # ]

    locality_inputs = {
        "neighborhood": {"prompt": locality_prompts, "ground_truth": locality_ans},
    }
    # portability_inputs = {
    #     "one_hop": {"prompt": portability_prompts, "ground_truth": portability_ans},
    # }
    # hparams = WISEHyperParams.from_hparams('./hparams/WISE/llama-7b.yaml')
    # hparams = WISEHyperParams.from_hparams("./hparams/WISE/gpt-j-6B.yaml")
    hparams = WISEHyperParams.from_hparams("./hparams/WISE/llama3-8b.yaml")

    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        target_new=target_new,
        rephrase_prompts=rephrase_prompts,
        locality_inputs=locality_inputs,
        loc_prompts=loc_prompts,  # 必须传入不相关样本（随机打乱的、来自训练集的不相关 Prompt 列表），用于训练路由器将编辑范围限制在最小空间，用于在训练时计算路由损失
        # portability_inputs=portability_inputs,
        sequential_edit=True,
    )

    return metrics, edited_model


# # Before editing

# logger.log("Before editing")

# model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
# logger.log("Model initialized")

# # output the response
# # evaluation metrics and questions
# Evaluation_metrics = ["Reliability", "Generalization", "Locality", "Portability"]
# Evaluation_prompts = [
#     "Who is the current President of the United States?",
#     "What is the name of the current President of the United States?",
#     "Where is the capital of the United States?",
#     "Where is the current U.S. President born ?",
# ]
# evaluate_chat_template(model, Evaluation_prompts, Evaluation_metrics, device=0)

# # clear memory
# del model
# torch.cuda.empty_cache()

# logger.log("Before editing memory cleared")

# # Start editing
# logger.log("Starting editing")

# ##  Edit once: Joe Biden ——> Donald Trump
# prompts = ["Who is the current President of the United States?"]
# subject = ["President"]
# ground_truth = ["Joe Biden"]
# target_new = ["Donald Trump"]


# # loc_prompts: used to provide xi in Equation 5 in the paper.
# loc_prompts = [
#     "nq question: ek veer ki ardaas veera meaning in english A Brother's Prayer... Veera"
# ]
# hparams = WISEHyperParams.from_hparams("./hparams/WISE/llama3.1-8b.yaml")
# editor = BaseEditor.from_hparams(hparams)
# metrics, edited_model, weights_copy = editor.edit(
#     prompts=prompts,
#     ground_truth=ground_truth,
#     target_new=target_new,
#     subject=subject,
#     loc_prompts=loc_prompts,
#     sequential_edit=True,
# )

# logger.log("Editing completed")

# # output the response
# evaluate_chat_template(
#     edited_model, Evaluation_prompts, Evaluation_metrics, device=hparams.device
# )

# # clear memory
# del edited_model, weights_copy, editor
# torch.cuda.empty_cache()
# logger.log("Memory cleared")


metrics, edited_model = test_WISE()
logger.log(f"Metrics: {metrics}")
logger.log("Editing completed")

# clear memory
del edited_model
torch.cuda.empty_cache()
logger.log("Memory cleared")

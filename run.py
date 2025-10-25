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

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from easyeditor import BaseEditor, WISEHyperParams
import torch
import torch.cuda

logger.log("Import dependencies finished")
# model_id = "meta-llama/Llama-3.1-8B"
model_id = "/data/user/yjiang717/data/model/Llama-3.1-8B"
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)
logger.log("Pipeline initialized")

logger.log("Generating sample response")
result = pipeline("Who is the current President of the United States?")
logger.log(f"Response generated: {result}")


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"


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


# Before editing

logger.log("Before editing")

model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
logger.log("Model initialized")

# output the response
# evaluation metrics and questions
Evaluation_metrics = ["Reliability", "Generalization", "Locality", "Portability"]
Evaluation_prompts = [
    "Who is the current President of the United States?",
    "What is the name of the current President of the United States?",
    "Where is the capital of the United States?",
    "Where is the current U.S. President born ?",
]
evaluate_chat_template(model, Evaluation_prompts, Evaluation_metrics, device=0)

# clear memory
del model
torch.cuda.empty_cache()

logger.log("Before editing memory cleared")


# Start editing
logger.log("Starting editing")

##  Edit once: Joe Biden ——> Donald Trump
prompts = ["Who is the current President of the United States?"]
subject = ["President"]
ground_truth = ["Joe Biden"]
target_new = ["Donald Trump"]


# loc_prompts: used to provide xi in Equation 5 in the paper.
loc_prompts = [
    "nq question: ek veer ki ardaas veera meaning in english A Brother's Prayer... Veera"
]
hparams = WISEHyperParams.from_hparams("./hparams/WISE/llama3.1-8b.yaml")
editor = BaseEditor.from_hparams(hparams)
metrics, edited_model, weights_copy = editor.edit(
    prompts=prompts,
    ground_truth=ground_truth,
    target_new=target_new,
    subject=subject,
    loc_prompts=loc_prompts,
    sequential_edit=True,
)

logger.log("Editing completed")

# output the response
evaluate_chat_template(
    edited_model, Evaluation_prompts, Evaluation_metrics, device=hparams.device
)

# clear memory
del edited_model, weights_copy, editor
torch.cuda.empty_cache()
logger.log("Memory cleared")

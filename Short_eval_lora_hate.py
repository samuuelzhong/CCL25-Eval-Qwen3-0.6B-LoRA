import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
import numpy as np
from scipy import stats


# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# 加载测试数据集
from datasets import load_dataset
full_dataset = load_dataset("json", data_files=r"D:\HydroLMM_PEFT\data\racism\annotated_test1.json")["train"]

# 加载模型
peft_directory = r'Qwen3-4B-LoRA-Racism'
model_name = "./dataroot/models/Qwen/Qwen3-4B"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=True
)

from peft import PeftModel

# 正确加载 LoRA 模型（需先加载基础模型）
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, peft_directory).to(device)
peft_model = peft_model.merge_and_unload()  # 关键步骤：合并 LoRA 权重


# 3. 生成响应
def generate_response(prompt, model):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.split("\n")[0].strip()  # 取第一个换行前的部分

examples = full_dataset
all_metrics = []
json_result = []
num_false_egt,  num_false_ep = 0, 0
for i, example in enumerate(examples):
    # 初始化
    prompt = example["instruction"] + "\n\n我需要你进行类似上述处理的句子是:" + example["content"] + "\n\n"
    print('Prompt：', prompt)
    gt = example['id']
    print('Gt：', gt)

    # 前向推理
    model_response = generate_response(prompt, peft_model)
    print('Qwen_answer：', model_response)
    result = {
        "prompt": prompt,          # 可选：保留输入指令
        "prediction": model_response,     # 模型预测值
        "ground_truth": gt   # 真实值
    }
    json_result.append(result)

    if i == 500:
        pass

with open("predictions_lora_racism.json", "w", encoding="utf-8") as f:
    json.dump(json_result, f, indent=4, ensure_ascii=False)


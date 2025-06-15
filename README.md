# CCL25-Eval-Qwen3-0.6B-LoRA

*自然语言处理课程项目*

## 项目概述
本项目基于通义千问Qwen3系列模型（0.6B/1.7B/4B参数版本），采用LoRA（Low-Rank Adaptation）微调技术，构建针对种族主义内容的检测与评估系统。包含完整的训练数据、微调脚本和评估工具链。

## 文件结构
```text
.
├── Qwen3-0.6B-LoRA-Racism/ # 0.6B参数模型微调结果
│ ├── adapter_config.json # 微调参数配置
│ ├── adapter_model.safetensors # 微调模型权重
│ ├── added_tokens.json # 添加的token
│ ├── config.json # Qwen3模型配置
│ ├── special_tokens_map.json # 特殊token的配置
│ ├── tokenizer.json # 分词器内容
│ ├── tokenizer_config.json # 分词器配置
│ ├── vocab.json # 词典文件
│ └── README.md # 模型说明
├── Qwen3-1.7B-LoRA-Racism/ # 1.7B参数版本
├── Qwen3-4B-LoRA-Racism/ # 4B参数版本
├── data/racism/ # 训练评估数据
│ ├── annotated_train.json # 添加任务描述的训练集
│ ├── annotated_test1.json # 添加任务描述的测试集1
│ ├── annotated_test2.json # 添加任务描述的训练集2
│ ├── annotated_test.json # 添加任务描述的测试集样本
│ ├── train.json # 原始训练集
│ ├── test1.json # 原始测试集1
│ └── test2.json # 原始测试集2
├── Short_eval_lora_hate.py # 测试集评估脚本
├── lora_FT_hate.py # LoRA微调主程序
├── model_download.py # Qwen3模型下载工具
├── json2txt.py # 评估数据格式转换
├── predictions_lora_racism_0.1671.json # 0.6B模型输出的json文件，比赛评估分数为0.1671
├── predictions_lora_racism_0.2238.json # 1.7B模型输出的json文件，比赛评估分数为0.2238
├── predictions_lora_racism_0.2775.json # 4B模型输出的json文件，比赛评估分数为0.2775
├── test0.6B_0.1671.txt # 经由predictions_lora_racism_0.1671.json转换，用于评估的txt文件
├── requirements.txt # 依赖文件
└── README.md # 本项目文档
```

## 运行
配置依赖
```python
pip install -r requirements.txt
```
运行Qwen3模型下载
```python
python model_download.py --repo_id Qwen\Qwen3-0.6B
```
运行lora微调Qwen3模型
```python
python lora_FT_hate.py
```
运行模型评估
```python
python Short_eval_lora_hate.py
```
转换评估数据格式
```python
python json2txt.py
```

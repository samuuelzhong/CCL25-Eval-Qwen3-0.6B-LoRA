import json


def extract_predictions_to_txt(input_json, output_txt):
    """
    从JSON文件中提取所有"prediction"字段内容并写入TXT文件
    :param input_json: 输入的JSON文件路径
    :param output_txt: 输出的TXT文件路径
    """
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        predictions = []
        for item in data:
            if 'prediction' in item:
                predictions.append(str(item['prediction']))

        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(predictions))

        print(f"成功提取 {len(predictions)} 条prediction数据到 {output_txt}")

    except FileNotFoundError:
        print("错误：输入的JSON文件不存在")
    except json.JSONDecodeError:
        print("错误：JSON文件格式不正确")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")


# 使用示例
extract_predictions_to_txt('predictions_lora_racism.json', 'test.txt')
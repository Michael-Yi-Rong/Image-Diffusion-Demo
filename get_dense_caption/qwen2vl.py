import os
import re
from openai import OpenAI
import base64
import json
from collections import defaultdict


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# change to your own path
folder_path = r"/SSD_DISK/datasets/nuscenes_mini/samples/CAM_FRONT"

pattern = r"(n\d{3}-\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}[-+]\d{4}__CAM_FRONT__)(\d+)\.jpg"

# 保存按前缀分组的文件
file_groups = defaultdict(list)

# 遍历文件夹中的文件并按前缀分组
for filename in os.listdir(folder_path):
    match = re.match(pattern, filename)
    if match:
        prefix = match.group(1)  # 提取文件名前缀
        file_groups[prefix].append(filename)  # 按前缀分组

print(len(file_groups.items()))

results = {}
output_json_path = r"/SSD_DISK/users/rongyi/projects/diffusion/get_dense_caption/dense_caption.json"

if os.path.exists(output_json_path):
    with open(output_json_path, "r") as json_file:
        existing_results = json.load(json_file)
else:
    existing_results = {}

for prefix, files in file_groups.items():
    print(prefix)

    # 如果该前缀已经处理过，跳过
    if prefix in existing_results:
        print(f"Skipping already processed prefix: {prefix}")
        continue

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 按照数字部分排序
    files.sort(key=lambda x: int(re.search(r"(\d+)", x).group(1)))

    # 构建视频序列的 Base64 数据
    files_urls = [
        f"data:image/jpeg;base64,{encode_image(os.path.join(folder_path, file))}" for file in files
    ]

    if len(files_urls) > 40:
        files_urls = files_urls[:40]
    # 视频的所有帧数的大小加起来不能超过12MB，否则会报错

    # video
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[{
            "role": "user",
            "content": [
                {"type": "video", "video": files_urls},  # 传递视频序列
                {"type": "text", "text": "Describe this scene in a dense caption method in one paragraph"}
            ]
        }]
    )

    results[prefix] = {
        "video": prefix,
        "description": completion.choices[0].message.content
    }
    print(results[prefix])

    with open(output_json_path, "w") as json_file:
        json.dump(results, json_file, indent=4)

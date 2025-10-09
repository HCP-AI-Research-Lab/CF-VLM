import os, json, torch
from PIL import Image
from datasets import Dataset
from diffusers import DiffusionPipeline
from tqdm import tqdm
from together import Together

# 配置
model_path = "Qwen2.5-VL-Finetune/Qwen/Qwen2___5-VL-7B-Instruct"
train_json_path = "Qwen2.5-VL-Finetune/coco_2014/data_vl.json"
cf_dir = "for_pipeline"
os.makedirs(cf_dir, exist_ok=True)

# 多个反事实句子生成 Prompt
cf_prompt_en = """
You are a counterfactual rewriting engine. Your task is to generate K distinct counterfactual versions of a given sentence, each modifying exactly one key element from the original.

Each counterfactual should:
- Change only one element, either:
  - an object attribute (e.g., color, size, position, quantity, category, etc.), or
  - a causal relationship (e.g., invert cause and effect)
- Be logically coherent, fluent, and visually descriptive
- Introduce a meaningful difference from the original
- Be different from the others (no repetitions)
- Use natural language only

⚠️ Format:
Please output the K counterfactual sentences in a numbered list:
1. [sentence #1]
2. [sentence #2]
...

Only return the list. Do not include the original sentence, explanations, or any additional text.
"""

# 初始化 Together 客户端
client = Together(api_key="")
MODEL_NAME = "Qwen/Qwen2-72B-Instruct"

# 调用模型生成文本
def generate_text_with_deepseekv3(prompt):
    try:
        response = client.chat.completions.create(
            extra_body={},
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": cf_prompt_en},
                {"role": "user", "content": prompt}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"生成文本时出错: {e}")
        return ""

# 提取编号反事实文本
def parse_numbered_outputs(text):
    lines = text.strip().splitlines()
    parsed = []
    for line in lines:
        if line.strip().startswith(tuple(f"{i}." for i in range(1, 21))):
            content = line.split(".", 1)[-1].strip()
            if content:
                parsed.append(content)
    return parsed

# 加载 Stable Diffusion 模型
def load_pipeline(cls, model_id, dtype, device="cuda"):
    try:
        pipe = cls.from_pretrained(model_id, torch_dtype=dtype, local_files_only=True).to(device)
        print(f"Loaded {model_id} from local cache.")
    except Exception as e:
        print(f"Local load failed for {model_id}, downloading... ({e})")
        pipe = cls.from_pretrained(model_id, torch_dtype=dtype).to(device)
    return pipe

def format_prompt(cf_text):
    return f"{cf_text}"

# 生成反事实数据（K个）
def gen_cf_data(K=4):
    print("🧠 正在加载训练数据集 ...")
    dataset = Dataset.from_json(train_json_path)

    print("🖼️ 正在加载 Stable Diffusion 模型 ...")
    sd_pipe = load_pipeline(DiffusionPipeline, "stabilityai/stable-diffusion-xl-base-1.0", torch.float16)

    for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="生成反事实样本", unit="样本"):
        conv = item["conversations"]
        orig_text = conv[0]["value"]
        orig_output = conv[1]["value"]
        prompt = f'Input: "{orig_output}"\nGenerate {K} counterfactual versions.'

        cf_text_block = generate_text_with_deepseekv3(prompt)
        cf_sentences = parse_numbered_outputs(cf_text_block)[:K]

        cf_samples = []
        for k, cf_text in enumerate(cf_sentences):
            cf_img = sd_pipe(format_prompt(cf_text), guidance_scale=7.5).images[0]
            cf_img_path = os.path.join(cf_dir, f"cf_img_{idx}_{k}.png")
            cf_img.save(cf_img_path)
            cf_samples.append({"cf_text": cf_text, "cf_image_path": cf_img_path})

        sample = {
            "orig_input": orig_text,
            "orig_output": orig_output,
            "cf_samples": cf_samples
        }
        with open(os.path.join(cf_dir, "cf_processed.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("🧹 正在释放 Stable Diffusion 模型 ...")
    del sd_pipe

if __name__ == "__main__":
    print("🚀 开始生成反事实数据 ...")
    gen_cf_data(K=4)

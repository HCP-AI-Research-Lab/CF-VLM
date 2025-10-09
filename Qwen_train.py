import os, json
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
from transformers import (
    AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from PIL import Image


# 配置
model_path = "Qwen2.5-VL-Finetune/Qwen/Qwen2___5-VL-7B-Instruct"
cf_dir = "Qwen2.5-VL-Finetune/cf_data"
os.makedirs(cf_dir, exist_ok=True)

# 加载模型
def load_qwen_model():
    print("🧠 正在加载 Qwen 基础模型中 ...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    base_model.enable_input_require_grads()
    return base_model


# 预处理函数
# Ensure processor is loaded at the beginning of the script
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def process_func(example):
    def extract_image_path(text):
        return text.split("<|vision_start|>")[1].split("<|vision_end|>")[0].strip()

    def encode(image_path, prompt_text, label_text, processor):  # Add processor as an argument
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": Image.open(image_path), "resized_height": 280, "resized_width": 280},
                {"type": "text", "text": "COCO Yes:"}
            ]
        }]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_input, vid_input = process_vision_info(messages)
        proc = processor(text=[prompt], images=img_input, videos=vid_input, return_tensors="pt", padding=True)
        label_tok = tokenizer(label_text.strip(), add_special_tokens=False)
        iid = proc.input_ids[0].tolist() + label_tok.input_ids + [tokenizer.pad_token_id]
        att = proc.attention_mask[0].tolist() + label_tok.attention_mask + [1]
        lab = [-100] * len(proc.input_ids[0]) + label_tok.input_ids + [tokenizer.pad_token_id]
        return iid[:8192], att[:8192], lab[:8192], proc.pixel_values, proc.image_grid_thw.squeeze(0)

    # 正样本图文对
    orig_input = example["orig_input"]
    orig_output = example["orig_output"]
    image_path = orig_input.split("<|vision_start|>")[1].split("<|vision_end|>")[0]
    iid, att, lab, pix, grid = encode(image_path, orig_input, orig_output, processor)

    # 反事实样本图文对
    cf_iids, cf_atts, cf_labs, cf_pixs, cf_grids = [], [], [], [], []
    cf_texts = [cf["cf_text"] for cf in example["cf_samples"]]
    cf_imgs = [cf["cf_image_path"] for cf in example["cf_samples"]]

    cf_ans = orig_output

    for cf_text, cf_img_path in zip(cf_texts, cf_imgs):
        iid_cf, att_cf, lab_cf, pix_cf, grid_cf = encode(cf_img_path, cf_text, cf_ans, processor)
        cf_iids.append(torch.tensor(iid_cf))
        cf_atts.append(torch.tensor(att_cf))
        cf_labs.append(torch.tensor(lab_cf))
        cf_pixs.append(pix_cf)
        cf_grids.append(grid_cf)

    return {
        "input_ids": torch.tensor(iid), "attention_mask": torch.tensor(att), "labels": torch.tensor(lab),
        "pixel_values": pix, "image_grid_thw": grid,
        "cf_input_ids_list": cf_iids,
        "cf_attention_mask_list": cf_atts,
        "cf_labels_list": cf_labs,
        "cf_pixel_values_list": cf_pixs,
        "cf_image_grid_thw_list": cf_grids
    }





# def process_func(example):
#     def encode(image_path, prompt_text, label_text, processor):  # Add processor as an argument
#         messages = [{
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": Image.open(image_path), "resized_height": 280, "resized_width": 280},
#                 {"type": "text", "text": "COCO Yes:"}
#             ]
#         }]
#         prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         img_input, vid_input = process_vision_info(messages)
#         proc = processor(text=[prompt], images=img_input, videos=vid_input, return_tensors="pt", padding=True)
#         label_tok = tokenizer(label_text.strip(), add_special_tokens=False)
#         iid = proc.input_ids[0].tolist() + label_tok.input_ids + [tokenizer.pad_token_id]
#         att = proc.attention_mask[0].tolist() + label_tok.attention_mask + [1]
#         lab = [-100] * len(proc.input_ids[0]) + label_tok.input_ids + [tokenizer.pad_token_id]
#         return iid[:8192], att[:8192], lab[:8192], proc.pixel_values, proc.image_grid_thw.squeeze(0)

#     # 正样本图文对
#     orig_input = example["orig_input"]
#     orig_output = example["orig_output"]
#     image_path = orig_input  # Directly assign the image path as it is in the new data format
#     iid, att, lab, pix, grid = encode(image_path, orig_input, orig_output, processor)

#     # 反事实样本图文对
#     cf_iids, cf_atts, cf_labs, cf_pixs, cf_grids = [], [], [], [], []
#     cf_texts = [cf["cf_text"] for cf in example["cf_samples"]]
#     cf_imgs = [cf["cf_image_path"] for cf in example["cf_samples"]]

#     cf_ans = orig_output

#     for cf_text, cf_img_path in zip(cf_texts, cf_imgs):
#         iid_cf, att_cf, lab_cf, pix_cf, grid_cf = encode(cf_img_path, cf_text, cf_ans, processor)
#         cf_iids.append(torch.tensor(iid_cf))
#         cf_atts.append(torch.tensor(att_cf))
#         cf_labs.append(torch.tensor(lab_cf))
#         cf_pixs.append(pix_cf)
#         cf_grids.append(grid_cf)

#     return {
#         "input_ids": torch.tensor(iid), "attention_mask": torch.tensor(att), "labels": torch.tensor(lab),
#         "pixel_values": pix, "image_grid_thw": grid,
#         "cf_input_ids_list": cf_iids,
#         "cf_attention_mask_list": cf_atts,
#         "cf_labels_list": cf_labs,
#         "cf_pixel_values_list": cf_pixs,
#         "cf_image_grid_thw_list": cf_grids
#     }



# 微调模型
class CausalVLModel(nn.Module):
    def __init__(self, base_model, lora_rank=16, alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        cfg = LoraConfig(task_type="CAUSAL_LM", r=lora_rank, lora_alpha=32,
                         target_modules=["q_proj", "k_proj", "v_proj"], lora_dropout=0.1, bias="none")
        self.lora = get_peft_model(base_model, cfg)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels,
                cf_input_ids_list, cf_attention_mask_list, cf_pixel_values_list, cf_image_grid_thw_list, cf_labels_list,
                intervene=None):
        B = input_ids.size(0)
        K = len(cf_input_ids_list[0])

        # 正例前向
        out = self.lora(input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw, labels=labels)

        # 反例前向
        all_cf_logits, all_cf_losses = [], []
        for k in range(K):
            input_ids_cf = torch.stack([cf_input_ids_list[i][k] for i in range(B)]).to(input_ids.device)
            att_cf = torch.stack([cf_attention_mask_list[i][k] for i in range(B)]).to(input_ids.device)
            pix_cf = torch.stack([cf_pixel_values_list[i][k] for i in range(B)]).to(input_ids.device)
            grid_cf = torch.stack([cf_image_grid_thw_list[i][k] for i in range(B)]).to(input_ids.device)
            lab_cf = torch.stack([cf_labels_list[i][k] for i in range(B)]).to(input_ids.device)

            out_cf = self.lora(input_ids=input_ids_cf, attention_mask=att_cf,
                                pixel_values=pix_cf, image_grid_thw=grid_cf, labels=lab_cf)
            all_cf_logits.append(out_cf.logits)
            all_cf_losses.append(out_cf.loss)

        logits_size = out.logits.size(1)
        all_cf_logits = [
            F.pad(cf_logit, (0, 0, 0, logits_size - cf_logit.size(1))) if cf_logit.size(1) < logits_size
            else cf_logit[:, :logits_size] for cf_logit in all_cf_logits
        ]

        all_image_logits = torch.cat([out.logits] + all_cf_logits, dim=0)
        image_feats = F.normalize(all_image_logits.mean(dim=1), dim=-1)
        text_feats = F.normalize(out.logits.mean(dim=1), dim=-1)

        sim_matrix = image_feats @ text_feats.T
        target = torch.zeros_like(sim_matrix)
        for i in range(B):
            target[i * (K + 1), i] = 1.0

        loss_clip = F.mse_loss(sim_matrix, target)
        loss_total = self.alpha * out.loss + self.beta * torch.stack(all_cf_losses).mean() + self.gamma * loss_clip

        return {"loss": loss_total}

    def gradient_checkpointing_enable(self, **kwargs):
        self.lora.gradient_checkpointing_enable(**kwargs)

    @property
    def config(self):
        return self.lora.config

    def save_pretrained(self, save_directory, **kwargs):
        self.lora.save_pretrained(save_directory, **kwargs)

        
if __name__ == "__main__":
    with open(os.path.join(cf_dir, "cf_processed.json"), "r") as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data).map(process_func)

    print("🧠 正在加载并微调Qwen模型 ...")
    base_model = load_qwen_model()
    model = CausalVLModel(base_model)
    model.lora.gradient_checkpointing_enable()

    args = TrainingArguments(
        output_dir="./output/causal_vl_flicker", per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_train_epochs=2, learning_rate=1e-4, save_steps=100, logging_steps=10,
        report_to="none", gradient_checkpointing=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    print("🎉 开始训练模型 ...")
    trainer.train()

    print("💾 正在保存微调后的模型 ...")
    save_dir = "./output/causal_vl/only_for_test"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"✅ 模型保存成功，路径：{save_dir}")

    print("🎉 训练完成，祝您学习愉快！")
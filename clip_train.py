import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import re
from torchvision.transforms.functional import to_tensor

# === 工具函数 ===
def clean_image_path(s):
    if "<|vision_start|>" in s and "<|vision_end|>" in s:
        return re.search(r"<\|vision_start\|>(.*?)<\|vision_end\|>", s).group(1)
    return s

# === 数据集 ===
class CounterfactualTripletDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        anchor_image_path = clean_image_path(item["orig_input"])
        anchor_image = Image.open(anchor_image_path).convert("RGB")
        anchor_text = item["orig_output"]
        cf_images = [Image.open(cf["cf_image_path"]).convert("RGB") for cf in item["cf_samples"]]
        cf_texts = [cf["cf_text"] for cf in item["cf_samples"]]
        return anchor_image, anchor_text, cf_images, cf_texts

from torchvision.transforms import Resize

def collate_fn(batch):
    anchor_imgs, anchor_txts, cf_img_lists, cf_txt_lists = zip(*batch)

    # 统一图像尺寸（例如 224x224，可按需调整）
    resize = Resize((224, 224))

    anchor_imgs = torch.stack([to_tensor(resize(img)) for img in anchor_imgs], dim=0)
    cf_imgs = [torch.stack([to_tensor(resize(cf_img)) for cf_img in cf_imgs_per_sample]) for cf_imgs_per_sample in cf_img_lists]

    return anchor_imgs, list(anchor_txts), cf_imgs, list(cf_txt_lists)

# === 模型定义 ===
class CausalContrastiveCLIP(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", alpha=1.0, beta=0.5, gamma=0.5, temperature=0.07):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature

    def forward(self, anchor_imgs, anchor_txts, cf_imgs_batch, cf_txts_batch):
        device = next(self.parameters()).device
        batch_size = anchor_imgs.size(0)

        # 原始正对 InfoNCE 对比
        inputs = self.processor(
            text=anchor_txts,
            images=[Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).astype('uint8')) for img in anchor_imgs],
            return_tensors="pt",
            padding=True,
            truncation=True,
            do_rescale=False  # ✅ 关键修改
        ).to(device)

        outputs = self.model(**inputs)
        img_embeds = F.normalize(outputs.image_embeds, dim=-1)
        txt_embeds = F.normalize(outputs.text_embeds, dim=-1)

        logits_i2t = (img_embeds @ txt_embeds.T) / self.temperature
        logits_t2i = (txt_embeds @ img_embeds.T) / self.temperature
        labels = torch.arange(batch_size).to(device)
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        loss_contrastive = 0.5 * (loss_i2t + loss_t2i)

        # 对称反事实 InfoNCE 对比
        total_negcl_loss = 0.0
        for i in range(batch_size):
            cf_texts = cf_txts_batch[i]
            cf_imgs = cf_imgs_batch[i]

            all_texts_1 = [anchor_txts[i]] + cf_texts
            all_images_1 = [anchor_imgs[i]] + cf_imgs
            images_1 = [Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).astype('uint8')) for img in all_images_1]

            inputs_1 = self.processor(
                text=all_texts_1,
                images=images_1,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            embeds_img_1 = F.normalize(self.model.get_image_features(pixel_values=inputs_1["pixel_values"]), dim=-1)
            embeds_txt_1 = F.normalize(self.model.get_text_features(input_ids=inputs_1["input_ids"], attention_mask=inputs_1["attention_mask"]), dim=-1)
            sim_matrix = (embeds_img_1 @ embeds_txt_1.T) / self.temperature
            loss_1 = F.cross_entropy(sim_matrix[0:1], torch.tensor([0], device=device)) + F.cross_entropy(sim_matrix[:,0:1].T, torch.tensor([0], device=device))

            all_texts_2 = cf_texts[:1] + [anchor_txts[i]]
            all_images_2 = cf_imgs[:1] + [anchor_imgs[i]]
            images_2 = [Image.fromarray((img.permute(1,2,0).cpu().numpy()*255).astype('uint8')) for img in all_images_2]

            inputs_2 = self.processor(
                text=all_texts_2,
                images=images_2,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            embeds_img_2 = F.normalize(self.model.get_image_features(pixel_values=inputs_2["pixel_values"]), dim=-1)
            embeds_txt_2 = F.normalize(self.model.get_text_features(input_ids=inputs_2["input_ids"], attention_mask=inputs_2["attention_mask"]), dim=-1)
            sim_matrix_2 = (embeds_img_2 @ embeds_txt_2.T) / self.temperature
            loss_2 = F.cross_entropy(sim_matrix_2[0:1], torch.tensor([0], device=device)) + F.cross_entropy(sim_matrix_2[:,0:1].T, torch.tensor([0], device=device))

            total_negcl_loss += 0.5 * (loss_1 + loss_2)

        loss_negcl_avg = total_negcl_loss / batch_size

        return self.alpha * loss_contrastive + self.beta * loss_negcl_avg

@torch.no_grad()
def predict_image_text_similarity(model, processor, image_path, text, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    image_embeds = F.normalize(outputs.image_embeds, dim=-1)
    text_embeds = F.normalize(outputs.text_embeds, dim=-1)
    return torch.sum(image_embeds * text_embeds).item()

def evaluate_on_subset(model, processor, device, test_json_path, subset_size=3000):
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_data = test_data[:subset_size]
    model.eval()
    correct = 0
    for item in test_data:
        image_path = item["conversations"][0]["value"]
        question_full = item["conversations"][1]["value"]
        gt_answer = item["conversations"][2]["value"].strip().upper()
        try:
            question_main, options = question_full.split("A:", 1)
            option_a, option_b = options.split("B:")
            prompt_a = question_main.strip() + " A: " + option_a.strip()
            prompt_b = question_main.strip() + " B: " + option_b.strip()
        except:
            prompt_a = question_full.strip() + " A"
            prompt_b = question_full.strip() + " B"
        sim_a = predict_image_text_similarity(model, processor, image_path, prompt_a, device)
        sim_b = predict_image_text_similarity(model, processor, image_path, prompt_b, device)
        if ("A" if sim_a > sim_b else "B") == gt_answer:
            correct += 1
    acc = correct / subset_size
    print(f"\n🧪 [评估] 前 {subset_size} 样本准确率: {acc:.2%}\n")
    model.train()

def train(model, dataloader, optimizer, device, eval_path, num_epochs=1):
    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for anchor_imgs, anchor_txts, cf_imgs, cf_txts in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            step += 1
            loss = model(anchor_imgs.to(device), anchor_txts, cf_imgs, cf_txts)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            if step % 400 == 0:
                print(f"📉 Step {step}: Train Loss = {loss.item():.4f}")
                evaluate_on_subset(model.model, model.processor, device, eval_path)
        print(f"✅ Epoch {epoch+1} 平均损失: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    train_data_path = "Qwen2.5-VL-Finetune/data_all/cf_merged.json"
    eval_data_path = "Qwen2.5-VL-Finetune/ConMe_data/att.json"
    with open(train_data_path, "r") as f:
        data = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalContrastiveCLIP().to(device)
    dataset = CounterfactualTripletDataset(data, model.processor)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)

    train(model, dataloader, optimizer, device, eval_path=eval_data_path, num_epochs=1)

    save_path = "./clip_causal_final"
    os.makedirs(save_path, exist_ok=True)
    model.model.save_pretrained(save_path)
    model.processor.save_pretrained(save_path)
    print(f"✅ 模型已保存至: {save_path}")

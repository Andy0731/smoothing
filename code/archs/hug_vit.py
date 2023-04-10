from transformers import ViTModel, ViTForImageClassification
import torch
from datasets import get_normalize_layer

# model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

def get_hug_model(arch):
    model = ViTForImageClassification.from_pretrained(arch)
    normalize_layer = get_normalize_layer('vitcf10')
    return torch.nn.Sequential(normalize_layer, model)

def get_hug_vit(arch):
    return ViTForImageClassification.from_pretrained(arch)
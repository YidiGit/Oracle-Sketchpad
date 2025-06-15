import os
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import open_clip
from tqdm import tqdm   # note: tqdm.notebook ➜ tqdm for normal scripts

# Set up the model
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
model.eval()

DATASET_DIR = "Data/dataset"
valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

all_embeddings, all_labels = [], []

class_names = sorted(
    d for d in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, d))
)

for lbl, cls in enumerate(tqdm(class_names, desc="classes")):
    class_dir = os.path.join(DATASET_DIR, cls)
    for fname in tqdm(os.listdir(class_dir), desc=cls, leave=False):
        if fname.startswith('.') or not fname.lower().endswith(tuple(valid_ext)):
            continue                                  # skip hidden/non-image files
        fpath = os.path.join(class_dir, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            img_t = preprocess(img).unsqueeze(0)
            with torch.no_grad():
                emb = model.encode_image(img_t).cpu().numpy()
            all_embeddings.append(emb)
            all_labels.append(lbl)
        except (UnidentifiedImageError, OSError) as e:
            print(f"⚠️  Skipped {fpath}: {e}")

np.save('animal_embeddings.npy', np.concatenate(all_embeddings, axis=0))
np.save('animal_labels.npy', np.array(all_labels, dtype=np.int32))
print(f"Done!  Saved {len(all_labels)} embeddings.")

# WARNING: %sh switch-cuda 11.7
# conda activate clip
# conda install pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pprint

import clip
import torch
from PIL import Image
from tqdm import tqdm

DRIVER_MAP = {
    'geordi': '2021_08_31_geordi_enyaq',
    'poli': '2021_09_06_poli_enyaq',
    'michal': '2021_11_05_michal_enyaq',
    'dans': '2021_11_18_dans_enyaq',
    'jakub': '2021_11_18_jakubh_enyaq',
}
DRIVER = 'geordi'
DATASET_NAME = f'2024-10-28-driver-all-frames/{DRIVER_MAP[DRIVER]}'
DATASET_DIR = Path().home() / f'source/driver-dataset/{DATASET_NAME}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {device}')

PREFIX = 'a photo of a person inside a car'
PROMPTS = {
    'normal': f'{PREFIX} with both hands on the steering wheel',
    'coughing': f'{PREFIX} coughing or holding a phone',
    #'phone': f'{PREFIX} holding a phone',
}
pprint(PROMPTS)

model, preprocess = clip.load('ViT-B/32', device=device)

text = clip.tokenize(list(PROMPTS.values())).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)

ANOMAL_IMAGES_PATHS = sorted((DATASET_DIR / 'anomal/images/').glob('*.jpg'))

batch_size = 128  # Adjust based on GPU memory
preds = []


# Function to preprocess a single image
def preprocess_image(image_path: Path) -> torch.Tensor:
    """Square crop and resize the image to 224x224."""
    image = Image.open(image_path)
    image = image.crop((0, 0, image.size[1], image.size[1]))
    # The `preprocess` function resizes to 224x224
    # and normalizes the image.
    return preprocess(image)


# Split ANOMAL_IMAGES into batches
for i in tqdm(range(0, len(ANOMAL_IMAGES_PATHS), batch_size)):
    batch = ANOMAL_IMAGES_PATHS[i : i + batch_size]

    # Use ThreadPoolExecutor to preprocess images in parallel
    with ThreadPoolExecutor() as executor:
        processed_images = list(executor.map(preprocess_image, batch))

    # Stack and move the preprocessed images to the device
    images = torch.stack(processed_images).to(device)

    with torch.no_grad():
        # Encode image batch
        image_features = model.encode_image(images)

        # Pass the image batch and text batch to the model
        logits_per_image, logits_per_text = model(
            images, text
        )  # Assuming text is preprocessed and batched if needed
        cls_batch = (
            logits_per_image.softmax(dim=-1).argmax(dim=-1).cpu().tolist()
        )  # Get classes for the batch

    preds.extend(cls_batch)  # Append batch predictions

# Save predictions as JSON
with open('preds_224.json', 'w') as f:
    json.dump(preds, f)

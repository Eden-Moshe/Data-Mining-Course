# Data-Mining-Course

# 🕵️ HitNet – Camouflaged Object Detection (Colab Guide)

This project runs the [HitNet](https://github.com/HUuxiaobin/HitNet) model for detecting camouflaged objects using the COD10K dataset.

---

## 📦 Requirements

- Google Colab
- Google Drive (to store dataset)
- Internet connection (for downloading model weights)

---

## 🖼️ 1. Download Dataset

Download the COD10K dataset from Kaggle:

🔗 [COD10K Dataset on Kaggle](https://www.kaggle.com/datasets/getcam/cod10k?resource=download)

1. Extract the ZIP on your local computer.
2. Upload the full extracted folder to your **Google Drive**.

---

## 🔗 2. Mount Google Drive in Colab

In your Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then access your dataset via:
```python
/content/drive/MyDrive/YourPathTo/COD10K-v3/
```

You may optionally copy it to Colab for faster access.

---

## 🔧 3. Clone the HitNet Repository

```bash
!git clone https://github.com/HUuxiaobin/HitNet.git
%cd /content/HitNet
```

---

## 📁 4. Prepare Model Files

Run the following block to prepare model weights:

```python
# Step into the correct working directory
%cd /content/HitNet

# Create folders
!mkdir -p model_pth pretrained_pvt

# Download the main HitNet model weights
!wget https://huggingface.co/stablediffusionuser/hitnet/resolve/main/Net_epoch_best.pth -O model_pth/Net_epoch_best.pth

# Create placeholder for backbone
!touch pretrained_pvt/pvt_v2_b2.pth

# Download pretrained PVTv2-B2 backbone
!wget https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth -O pretrained_pvt/pvt_v2_b2.pth
```

---

## 🛠️ 5. Runtime Reset? Fix Imports

If Colab resets and the repo is lost:

```python
!git clone https://github.com/HUuxiaobin/HitNet.git
%cd /content/HitNet
import sys
sys.path.append('/content/HitNet')
```

---

## 🤖 6. Load the Model

```python
from lib.pvt import Hitnet
import torch

model = Hitnet()
model.load_state_dict(torch.load("model_pth/Net_epoch_best.pth", map_location='cpu'))
model.eval()
```

---

## 📷 7. Run Prediction on a Single Image

```python
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Load image (pick from Test set)
image_path = "/content/drive/MyDrive/.../COD10K-v3/Test/Image/your_image.jpg"
img = Image.open(image_path).convert("RGB")

# Preprocess
transform = transforms.Compose([
    transforms.Resize((352, 352)),
    transforms.ToTensor()
])
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    pred = model(input_tensor)[0][0].cpu().numpy().squeeze()

# Show
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Input Image")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(pred, cmap='gray')
plt.title("HitNet Prediction")
plt.axis('off')
plt.show()
```

---

## 📝 Notes
- if you want to run the notebook with the images shown in the presentation, we added a file named images put it in content/ as shown in the last cell and run it 
- You can select different images by changing the `image_path`.
- For batch inference or saving masks, loop through images and use `Image.fromarray(...)`.

---

## ✅ You're Done!

You now have a working HitNet environment in Google Colab with a full pipeline from dataset to prediction.

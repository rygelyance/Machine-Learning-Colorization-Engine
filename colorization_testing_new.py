import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# ======= Load TorchScript Model =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NUMBER = 1
scripted_model = torch.jit.load(f"model_{MODEL_NUMBER}.pt", map_location=device)
scripted_model.eval()

# Print TorchScript model info
print("\n=== TorchScript Model Structure ===\n")
print(scripted_model)

# ======= Preprocess Grayscale Image =======
def preprocess_grayscale_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_np = np.array(img) / 255.0
    lab = color.rgb2lab(img_np).astype("float32")
    L = lab[:, :, 0:1] / 50.0 - 1.0  # Normalize to [-1, 1]
    L_tensor = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return L_tensor, lab[:, :, 0]

# ======= Postprocess Output =======
def postprocess_output(L_orig, ab_pred, boost_saturation=True, shift_b=False, shift_a=False):
    # ab_pred from TorchScript model
    ab = ab_pred[0].cpu().numpy().transpose(1, 2, 0) * 128.0
    if boost_saturation:
        ab *= 1.8
    if shift_b:
        ab[:, :, 1] -= 5
    if shift_a:
        ab[:, :, 0] += 5
    lab = np.concatenate((L_orig[:, :, np.newaxis].astype("float32"), ab), axis=2)
    rgb = color.lab2rgb(lab)
    return np.clip(rgb, 0, 1)

# ======= Run Inference and Display =======
def colorize_image(img_path, save=False, save_path=None):
    L_tensor, L_orig = preprocess_grayscale_image(img_path)
    with torch.no_grad():
        ab_pred = scripted_model(L_tensor)
    rgb_result = postprocess_output(L_orig, ab_pred)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Grayscale Input")
    plt.imshow(L_orig, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Colorized Output")
    plt.imshow(rgb_result)
    plt.axis('off')

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
        print(f"Saved output to {save_path}")
    else:
        plt.show()

# ======= Test Images =======
test_images = [
    "GS_Test_Images/DoubleDecker.jpg",
    "GS_Test_Images/UMD.jpg",
    "GS_Test_Images/Bonsai.jpg",
    "GS_Test_Images/Study.jpg",
    "GS_Test_Images/Artemesia.jpg",
    "GS_Test_Images/Me.jpg"
]

for img in test_images:
    colorize_image(img)

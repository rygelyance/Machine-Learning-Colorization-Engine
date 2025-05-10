import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from PIL import Image
from torch import nn
from colorization_training import UNetColorization

# ======= Load the Model =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNetColorization().to(device)
model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
model.eval()

# ======= Preprocess Grayscale Image =======
def preprocess_grayscale_image(img_path):
    img = Image.open(img_path).convert('RGB').resize((256, 256))
    img_np = np.array(img) / 255.0
    lab = color.rgb2lab(img_np).astype("float32")
    L = lab[:, :, 0:1] / 50.0 - 1.0  # Normalize to [-1, 1]
    L_tensor = torch.tensor(L).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return L_tensor, lab[:, :, 0]

# ======= Reconstruct Color Image =======
def postprocess_output(L_orig, ab_pred):
    ab_pred = ab_pred.squeeze().cpu().detach().numpy()
    ab_pred = ab_pred.transpose(1, 2, 0)
    ab_pred *= 128.0

    # Post-processing result to artificially boost saturation (the band-aid fix)
    ab_pred *= 1.8

    # Shift color temperature: decrease 'b' (yellow → blue), optionally 'a' (red → green) (the other band-aid fix)
    ab_pred[:, :, 1] -= 5  # Shift toward blue
    # ab_pred[:, :, 0] += 5  # Uncomment to adjust red/green

    L_orig = L_orig.astype("float32")
    lab = np.concatenate((L_orig[:, :, np.newaxis], ab_pred), axis=2)
    rgb = color.lab2rgb(lab)
    return np.clip(rgb, 0, 1)

# ======= Run Inference =======
def colorize_image(img_path):
    L_tensor, L_orig = preprocess_grayscale_image(img_path)
    with torch.no_grad():
        ab_pred = model(L_tensor)
    rgb_result = postprocess_output(L_orig, ab_pred)

    # Display original grayscale and colorized image
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
    plt.show()

# ======= Run on an Image =======
colorize_image("GS_Test_Images/DoubleDecker.jpg")
colorize_image("GS_Test_Images/UMD.jpg")
colorize_image("GS_Test_Images/Bonsai.jpg")
colorize_image("GS_Test_Images/Study.jpg")
colorize_image("GS_Test_Images/Artemesia.jpg")
colorize_image("GS_Test_Images/Me.jpg")
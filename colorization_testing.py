import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from skimage import color
from skimage.transform import resize

# ======= Load TorchScript Model =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model Directory:
# model_1.pt - Trained on 5K images from COCO, 5 epochs
# model_2.pt - Trained on 1,668 images of people from Kaggle dataset, 5 epochs
# model_3.pt - Trained on 4,300 images of landscapes from Kaggle dataset, 5 epochs
MODEL_NUMBER = 3
scripted_model = torch.jit.load(f"model_{MODEL_NUMBER}.pt", map_location=device)
scripted_model.eval()

# Print TorchScript model info
print("\n=== TorchScript Model Structure ===\n")
print(scripted_model)

# ======= Preprocess Grayscale Image =======
def preprocess_grayscale_image(img_path):
    img = Image.open(img_path).convert('RGB')
    orig_size = img.size  # (width, height)

    img_resized = img.resize((256, 256))

    img_np = np.array(img_resized) / 255.0
    lab = color.rgb2lab(img_np).astype("float32")
    L = lab[:, :, 0:1] / 50.0 - 1.0  # Normalize to [-1, 1]
    L_tensor = torch.from_numpy(L).permute(2, 0, 1).unsqueeze(0).float().to(device)


    return L_tensor, lab[:, :, 0], orig_size

# ======= Postprocess Output =======
def postprocess_output(L_orig, ab_pred, orig_size):
    # ab_pred from TorchScript model
    ab = ab_pred[0].cpu().numpy().transpose(1, 2, 0) * 128.0

    # Saturation and color temp adjustments
    ab *= 1.6
    # ab[:, :, 1] -= 10
    # ab[:, :, 0] += 5

    lab = np.concatenate((L_orig[:, :, np.newaxis].astype("float32"), ab), axis=2)
    rgb = color.lab2rgb(lab)
    rgb_clipped = np.clip(rgb, 0, 1)

    # Resize back to original image size (width, height)
    h, w = orig_size[1], orig_size[0]
    rgb_upscaled = resize(rgb_clipped, (h, w), mode='reflect', anti_aliasing=True)

    return rgb_upscaled

# ======= Run Inference and Display =======
def colorize_image(img_path, save=False, save_path=None):
    L_tensor, L_orig, orig_size = preprocess_grayscale_image(img_path)
    with torch.no_grad():
        ab_pred = scripted_model(L_tensor)
    rgb_result = postprocess_output(L_orig, ab_pred, orig_size)

    # Resize grayscale L_orig to original size for comparision
    L_resized = resize(L_orig, (orig_size[1], orig_size[0]), mode='reflect', anti_aliasing=True)


    # Plot the grayscale, colorized, and original images
    plt.figure(figsize=(15, 5))  # Wider to fit three images
    plt.subplot(1, 3, 1)
    plt.title("Grayscale Input")
    plt.imshow(L_resized, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Colorized Output")
    plt.imshow(rgb_result)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Original Color Image")
    plt.imshow(mpimg.imread(img_path))
    plt.axis('off')

    plt.tight_layout()
    if save and save_path:
        plt.savefig(save_path)
        print(f"Saved output to {save_path}")
    else:
        plt.show()

# ======= Test Images Directory =======
# Test_Images of Landscapes and Whatnot, Human_Test_Images of human subjects from stock photos
test_images_dir = "Landscape_Test_Images"

# Get all image file paths in the directory
test_images = [os.path.join(test_images_dir, fname)
               for fname in os.listdir(test_images_dir)
               if fname.lower().endswith((".jpg", ".jpeg", ".png"))]

# Run colorization on each image
for img_path in sorted(test_images):
    print(f"Processing: {img_path}")
    colorize_image(img_path)

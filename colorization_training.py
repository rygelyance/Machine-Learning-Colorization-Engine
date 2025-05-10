import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage import color
from torch import nn, optim

print(torch.__version__)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======= Model Definition =======
class UNetColorization(nn.Module):
    def __init__(self):
        super(UNetColorization, self).__init__()

        # Helper function to define a double convolutional block (Conv → ReLU → Conv → ReLU)
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        # -------- Encoder (Downsampling path) --------
        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # -------- Bottleneck --------
        self.bottleneck = conv_block(256, 512)

        # -------- Decoder (Upsampling path) --------
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        # Final output layer: map 64 features → 2 channels (ab color space)
        self.final = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return out

# ======= Dataset Class =======
class CocoColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB').resize((256, 256))
        img_np = np.array(img) / 255.0

        # Convert RGB to LAB
        lab = color.rgb2lab(img_np).astype("float32")
        L = lab[:, :, 0:1] / 50.0 - 1.0       # Normalize to [-1, 1]
        ab = lab[:, :, 1:] / 128.0            # Normalize to [-1, 1]

        if self.transform:
            L = self.transform(L)
            ab = self.transform(ab)

        return torch.tensor(L).permute(2, 0, 1), torch.tensor(ab).permute(2, 0, 1)
    
# ======= Training Function =======
def train(model, dataloader, epochs=10, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # Possibly try SmoothL1Loss or other functions??

    for epoch in range(epochs):
        print(f"Training Epoch {epoch}")
        total_loss = 0
        for L, ab in dataloader:
            L, ab = L.cuda(device), ab.cuda(device)
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# ======= Entry Point =======
if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Initialize dataset (Utilizing 5000 images from the 2017 Tiny COCO Subset)
    dataset = CocoColorizationDataset(root_dir='COCO_Images/val2017')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Model setup
    model = UNetColorization().to(device)

    # Train model
    train(model, dataloader, epochs=20)

    # Save trained model
    torch.save(model.state_dict(), "colorization_model.pth")
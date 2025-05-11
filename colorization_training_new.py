import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage import color
from torch import nn, optim
from tqdm import tqdm  # Progress bar

# ======= Residual Block =======
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# ======= UNet Colorization with Residual Blocks =======
class UNetColorization(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ResidualBlock(out_ch)
            )

        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            conv_block(64, 128),
            ResidualBlock(128)
        )

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)

        self.final = nn.Conv2d(32, 2, kernel_size=1)
        self.final_activation = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final_activation(self.final(d1))
        return out

# ======= Dataset Class =======
class CocoColorizationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB').resize((256, 256))
        img_np = np.array(img) / 255.0
        lab = color.rgb2lab(img_np).astype("float32")
        L = lab[:, :, 0:1] / 50.0 - 1.0
        ab = lab[:, :, 1:] / 128.0
        if self.transform:
            L = self.transform(L)
            ab = self.transform(ab)
        return torch.from_numpy(L).permute(2, 0, 1), torch.from_numpy(ab).permute(2, 0, 1)

# ======= Training Function =======
def train(model, dataloader, epochs=10, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Training {epoch+1}", leave=False)
        for L, ab in pbar:
            L, ab = L.to(device), ab.to(device)
            optimizer.zero_grad()
            pred_ab = model(L)
            loss = criterion(pred_ab, ab)
            loss.backward()
            optimizer.step()
            batch_loss = loss.item()
            total_loss += batch_loss

            pbar.set_postfix(batch_loss=f"{batch_loss:.4f}")
        print(f"  Epoch Loss: {total_loss/len(dataloader):.4f}")

# ======= Entry Point =======
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Prepare data
    dataset = CocoColorizationDataset(root_dir='COCO_Images/val2017')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = UNetColorization().to(device)

    # TorchScript compile before training and print
    example_input = torch.randn(1, 1, 256, 256).to(device)
    traced = torch.jit.trace(model, example_input)
    print("Compiled TorchScript Model:\n", traced)

    # Train model
    train(model, dataloader, epochs=5, lr=1e-3)

    # Save model with incrementing filename
    def get_next_model_filename(base="model", ext=".pt"):
        i = 1
        while os.path.exists(f"{base}_{i}{ext}"):
            i += 1
        return f"{base}_{i}{ext}"

    model.eval()
    scripted = torch.jit.script(model)
    filename = get_next_model_filename()
    scripted.save(filename)
    print(f"Saved TorchScript model to {filename}")

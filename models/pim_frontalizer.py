import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowGenerator_GS(nn.Module):
    """
    Shallow Generator (G_S) from D2SC-GAN (Bhattacharjee & Das, IEEE T-BIOM 2020, Fig. 2).
    Captures low-frequency components, contrast, and macro-structural face features.
    
    Architecture (Fig. 2):
    - CONV_2D_1 [(1, 1), #3] -> FLATTEN -> DENSE_1 [2048] -> DENSE_2 [35 x 35 x 128] -> RESHAPE
    - UPSAMPLING_2D_1 [2, 2] -> CONV_2D_2 [(5, 5), #64]
    - UPSAMPLING_2D_2 [2, 2] -> CONV_2D_3 [(5, 5), #64] -> CONV_2D_4 [(5, 5), #3]
    """
    def __init__(self, input_size=(112, 112)):
        super(ShallowGenerator_GS, self).__init__()
        self.input_size = input_size
        
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.dense1 = nn.Linear(3 * input_size[0] * input_size[1], 2048)
        self.dense2 = nn.Linear(2048, 35 * 35 * 128)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h1 = self.relu(self.conv1(x))
        h1_flat = h1.view(B, -1)
        
        d1 = self.relu(self.dense1(h1_flat))
        d2 = self.relu(self.dense2(d1))
        
        reshaped = d2.view(B, 128, 35, 35)
        
        up1 = F.interpolate(reshaped, scale_factor=2, mode='nearest') # 70x70
        h2 = self.relu(self.conv2(up1))
        
        up2 = F.interpolate(h2, scale_factor=2, mode='nearest')      # 140x140
        h3 = self.relu(self.conv3(up2))
        
        out = torch.tanh(self.conv4(h3))
        if out.shape[2:] != self.input_size:
            out = F.interpolate(out, size=self.input_size, mode='bilinear', align_corners=False)
        return out

class DeepGenerator_GD(nn.Module):
    """
    Deep Generator (G_D) from D2SC-GAN (Bhattacharjee & Das, IEEE T-BIOM 2020, Fig. 3).
    Captures high-frequency fine details, crisp edges, and identity-discriminative facial texture.
    
    Architecture (Fig. 3):
    - CONV_2D_1 [(1, 1), #3] -> FLATTEN -> DENSE_1 [2048] -> DENSE_2 [35 x 35 x 128] -> RESHAPE
    - UPSAMPLING_2D_1 [2, 2] -> CONV_2D_2 [(5, 5), #128]
    - UPSAMPLING_2D_2 [2, 2] -> CONV_2D_3 [(5, 5), #256] -> CONV_2D_4 [(5, 5), #128]
      -> CONV_2D_5 [(5, 5), #64] -> CONV_2D_6 [(5, 5), #32] -> CONV_2D_7 [(5, 5), #3]
    """
    def __init__(self, input_size=(112, 112)):
        super(DeepGenerator_GD, self).__init__()
        self.input_size = input_size
        
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.dense1 = nn.Linear(3 * input_size[0] * input_size[1], 2048)
        self.dense2 = nn.Linear(2048, 35 * 35 * 128)
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv7 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h1 = self.relu(self.conv1(x))
        h1_flat = h1.view(B, -1)
        
        d1 = self.relu(self.dense1(h1_flat))
        d2 = self.relu(self.dense2(d1))
        
        reshaped = d2.view(B, 128, 35, 35)
        
        up1 = F.interpolate(reshaped, scale_factor=2, mode='nearest') # 70x70
        h2 = self.relu(self.conv2(up1))
        
        up2 = F.interpolate(h2, scale_factor=2, mode='nearest')      # 140x140
        h3 = self.relu(self.conv3(up2))
        h4 = self.relu(self.conv4(h3))
        h5 = self.relu(self.conv5(h4))
        h6 = self.relu(self.conv6(h5))
        
        out = torch.tanh(self.conv7(h6))
        if out.shape[2:] != self.input_size:
            out = F.interpolate(out, size=self.input_size, mode='bilinear', align_corners=False)
        return out

class D2SCGANDiscriminator(nn.Module):
    """
    Discriminator (D) from D2SC-GAN (Bhattacharjee & Das, IEEE T-BIOM 2020, Fig. 4).
    Dual-Head Architecture:
    - Branch 1 (D_bin): Real vs Fake Adversarial Discriminator Head.
    - Branch 2 (D_cls): Multi-Class Closed-Set Identity Classification Head.
    
    Architecture (Fig. 4):
    - CONV_2D_1 [(5, 5), #64] -> MAXPOOLING_2D_1 [2, 2]
    - CONV_2D_2 [(5, 5), #128] -> MAXPOOLING_2D_2 [2, 2] -> FLATTEN -> DENSE_1 [1024]
    - Split: D_bin (DENSE_2 [1] + Sigmoid) and D_cls (DENSE_3 [C] + Softmax)
    """
    def __init__(self, num_classes: int = 100, input_size=(112, 112)):
        super(D2SCGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Feature map size after 2 maxpools: (112 / 4) = 28 -> 28 x 28 x 128
        h_feat, w_feat = input_size[0] // 4, input_size[1] // 4
        self.dense1 = nn.Linear(128 * h_feat * w_feat, 1024)
        
        # Dual Branch Heads
        self.d_bin = nn.Linear(1024, 1)          # Real / Fake logits
        self.d_cls = nn.Linear(1024, num_classes) # Multi-class identity logits

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        h1 = self.pool1(self.leaky_relu(self.conv1(x)))
        h2 = self.pool2(self.leaky_relu(self.conv2(h1)))
        
        h2_flat = h2.view(B, -1)
        feat_1024 = self.leaky_relu(self.dense1(h2_flat))
        
        out_bin = torch.sigmoid(self.d_bin(feat_1024))
        out_cls = self.d_cls(feat_1024)
        return out_bin, out_cls

class D2SCGANHybridLoss(nn.Module):
    """
    Official D2SC-GAN Multi-Objective Hybrid Loss Function (IEEE T-BIOM 2020, Section III):
    1. Multi-Resolution Patch-wise MSE (MR_PMSE): Multi-scale spatial patch MSE over 1x1, 2x2, 4x4 grids.
    2. Normalized Chi-Squared Distance (NCD): Chi-squared histogram distance between real and generated images.
    3. Kullback-Leibler Divergence (KLD): Maximize KLD (minimize negative KLD) between G_S and G_D feature distributions.
    4. Adversarial Loss (BCE) + Identity Classification Loss (CE).
    """
    def __init__(self, lambda_mr: float = 1.0, lambda_ncd: float = 0.5, lambda_kld: float = 0.1):
        super(D2SCGANHybridLoss, self).__init__()
        self.lambda_mr = lambda_mr
        self.lambda_ncd = lambda_ncd
        self.lambda_kld = lambda_kld
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def patchwise_mse(self, gen: torch.Tensor, real: torch.Tensor, grid_size: int) -> torch.Tensor:
        B, C, H, W = gen.size()
        h_patch, w_patch = H // grid_size, W // grid_size
        total_mse = 0.0
        for i in range(grid_size):
            for j in range(grid_size):
                g_patch = gen[:, :, i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch]
                r_patch = real[:, :, i*h_patch:(i+1)*h_patch, j*w_patch:(j+1)*w_patch]
                total_mse += F.mse_loss(g_patch, r_patch)
        return total_mse / (grid_size * grid_size)

    def mr_pmse_loss(self, gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        mse_1 = F.mse_loss(gen, real)
        mse_2 = self.patchwise_mse(gen, real, grid_size=2)
        mse_4 = self.patchwise_mse(gen, real, grid_size=4)
        return (mse_1 + mse_2 + mse_4) / 3.0

    def ncd_loss(self, gen: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        gen_pos = torch.relu(gen) + 1e-7
        real_pos = torch.relu(real) + 1e-7
        chi_sq = torch.sum(((gen_pos - real_pos) ** 2) / (gen_pos + real_pos + 1e-7))
        return chi_sq / gen.numel()

    def kld_loss(self, shallow_out: torch.Tensor, deep_out: torch.Tensor) -> torch.Tensor:
        p_shallow = F.softmax(shallow_out.view(shallow_out.size(0), -1), dim=-1) + 1e-7
        p_deep = F.softmax(deep_out.view(deep_out.size(0), -1), dim=-1) + 1e-7
        # Negative KLD minimization (ensures G_S and G_D capture distinct frequency bands)
        kld = torch.sum(p_shallow * torch.log(p_shallow / p_deep), dim=-1)
        return kld.mean()

    def forward(self, gen_shallow: torch.Tensor, gen_deep: torch.Tensor, real: torch.Tensor, d_bin_out: torch.Tensor = None, d_cls_out: torch.Tensor = None, labels: torch.Tensor = None):
        rec_s = self.mr_pmse_loss(gen_shallow, real) + self.lambda_ncd * self.ncd_loss(gen_shallow, real)
        rec_d = self.mr_pmse_loss(gen_deep, real) + self.lambda_ncd * self.ncd_loss(gen_deep, real)
        
        kld = self.kld_loss(gen_shallow, gen_deep)
        total_loss = self.lambda_mr * (rec_s + rec_d) - self.lambda_kld * kld
        
        if d_bin_out is not None:
            real_targets = torch.ones_like(d_bin_out)
            total_loss += self.bce_loss(d_bin_out, real_targets)
        if d_cls_out is not None and labels is not None:
            total_loss += self.ce_loss(d_cls_out, labels)
            
        return total_loss

class D2SCGANSuperRes(nn.Module):
    """
    Dual Deep-Shallow Channeled GAN (D2SC-GAN) for Low-Resolution Faces in the Wild.
    Integrates Shallow Generator G_S and Deep Generator G_D.
    """
    def __init__(self, input_size=(112, 112)):
        super(D2SCGANSuperRes, self).__init__()
        self.g_shallow = ShallowGenerator_GS(input_size=input_size)
        self.g_deep = DeepGenerator_GD(input_size=input_size)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_s = self.g_shallow(x)
        out_d = self.g_deep(x)
        concat_out = torch.cat([out_s, out_d], dim=1)
        sr_img = self.fusion_conv(concat_out)
        return sr_img

class PIMFrontalizationGAN(nn.Module):
    """
    Pose-Invariant Model (PIM) Frontalization Network combined with D2SC-GAN Super Resolution.
    Synthesizes a canonical frontal, crisp-illuminated face view from an extreme profile/yaw rotated face crop.
    """
    def __init__(self, input_size=(112, 112)):
        super(PIMFrontalizationGAN, self).__init__()
        self.super_res = D2SCGANSuperRes(input_size=input_size)
        
        # Encoder: compresses profile image into latent pose-free identity code z
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 14x14
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Symmetric Decoder: synthesizes frontalized canonical face
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),   # 112x112
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, yaw_angle: float = 0.0) -> torch.Tensor:
        if abs(yaw_angle) < 20.0:
            return x
            
        sr_x = self.super_res(x)
        z = self.encoder(sr_x)
        frontalized_x = self.decoder(z)
        
        flipped_x = torch.flip(frontalized_x, dims=[3])
        sym_frontalized = 0.5 * (frontalized_x + flipped_x)
        return sym_frontalized

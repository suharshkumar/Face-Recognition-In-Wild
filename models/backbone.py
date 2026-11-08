import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'IResNet', 'IBasicBlock', 'iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200',
    'FaceVisionTransformer', 'vit_face_base', 'vit_face_large', 'vit_face_huge',
    'FasterRCNNBackbone', 'fasterrcnn_backbone', 'ResNet100Backbone'
]

# =====================================================================
# 1. Standard InsightFace / ArcFace / CurricularFace IResNet Backbone
# =====================================================================

class IBasicBlock(nn.Module):
    """
    Standard InsightFace / ArcFace / CurricularFace Improved Basic Block (IBasicBlock).
    Structure: BN1 -> Conv3x3 (s=1) -> BN2 -> PReLU -> Conv3x3 (s=stride) -> BN3 + Residual
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample=None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('IBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in IBasicBlock")

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class IResNet(nn.Module):
    """
    Standard InsightFace / ArcFace / CurricularFace Improved ResNet (IResNet) Architecture.
    Matches deepinsight/insightface official reference implementation.
    """
    fc_scale = 7 * 7

    def __init__(self,
                 block,
                 layers,
                 dropout: float = 0.4,
                 num_features: int = 512,
                 zero_init_residual: bool = False,
                 groups: int = 1,
                 width_per_group: int = 64,
                 replace_stride_with_dilation=None,
                 fp16: bool = False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or 3-tuple, got {replace_stride_with_dilation}")
            
        self.groups = groups
        self.base_width = width_per_group
        
        # Stem: Conv 3 -> 64 (3x3, stride=1, padding=1, bias=False) -> BN1 -> PReLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
        self.prelu = nn.PReLU(64)
        
        # Four Residual Stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        # Head: BN2 -> Dropout -> Flatten -> Linear -> BN3
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None
        self.fc = nn.Linear(512 * self.fc_scale, num_features, bias=False)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        
        # Standard InsightFace weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1, dilate: bool = False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation)
            )

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.features(x)
        
        return F.normalize(x, p=2, dim=1)

# =====================================================================
# 2. Vision Transformer (ViT-Face / TransFace) Backbone Architecture
# =====================================================================

class ViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True):
        super(ViTAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FaceVisionTransformer(nn.Module):
    """
    SOTA Vision Transformer Backbone for Deep Face Recognition (TransFace / ViT-Face).
    Leverages global multi-head self-attention to dynamically route features around occluded face zones.
    """
    def __init__(self, img_size: int = 112, patch_size: int = 8, in_chans: int = 3,
                 num_features: int = 512, embed_dim: int = 512, depth: int = 12,
                 num_heads: int = 8, dropout: float = 0.1, **kwargs):
        super(FaceVisionTransformer, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList([
            ViTBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # 512-dim L2-normalized feature projection head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_features, bias=False),
            nn.BatchNorm1d(num_features)
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_feat = x[:, 0] # Extract CLS token feature
        
        out = self.head(cls_feat)
        return F.normalize(out, p=2, dim=1)

# Factory functions
def iresnet18(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [2, 2, 2, 2], dropout=dropout, num_features=num_features, **kwargs)

def iresnet34(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [3, 4, 6, 3], dropout=dropout, num_features=num_features, **kwargs)

def iresnet50(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [3, 4, 14, 3], dropout=dropout, num_features=num_features, **kwargs)

def iresnet100(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [3, 13, 30, 3], dropout=dropout, num_features=num_features, **kwargs)

def iresnet200(dropout=0.4, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return IResNet(IBasicBlock, [6, 26, 60, 6], dropout=dropout, num_features=num_features, **kwargs)

def vit_face_base(dropout=0.1, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return FaceVisionTransformer(img_size=112, patch_size=8, embed_dim=512, depth=12, num_heads=8, num_features=num_features, dropout=dropout, **kwargs)

def vit_face_large(dropout=0.1, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return FaceVisionTransformer(img_size=112, patch_size=8, embed_dim=768, depth=24, num_heads=12, num_features=num_features, dropout=dropout, **kwargs)

def vit_face_huge(dropout=0.1, num_features=512, embedding_dim=None, **kwargs):
    if embedding_dim is not None:
        num_features = embedding_dim
    return FaceVisionTransformer(img_size=112, patch_size=8, embed_dim=1024, depth=32, num_heads=16, num_features=num_features, dropout=dropout, **kwargs)

class FasterRCNNBackbone(nn.Module):
    """
    Pretrained Faster-RCNN ResNet-50 FPN Deep Feature Representation Backbone.
    
    Extracts intermediate multi-scale feature representations (p2, p3, p4, p5 feature maps)
    from Faster-RCNN before the final classification/box prediction layer, fusing them
    into a 512-dimensional L2-normalized identity embedding vector f in R^512.
    """
    def __init__(self, embedding_dim: int = 512):
        super(FasterRCNNBackbone, self).__init__()
        self.embedding_dim = embedding_dim
        
        try:
            import torchvision.models.detection as detection
            faster_rcnn = detection.fasterrcnn_resnet50_fpn(weights=detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            self.fpn_backbone = faster_rcnn.backbone # ResNet-50 FPN feature representation layers before output layer
        except Exception:
            self.fpn_backbone = None

        # Registered, deterministic fallback. The old implementation created a
        # new random Linear layer inside every forward call, so it could never
        # learn and produced different embeddings for identical inputs.
        self.fallback_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(3 * 7 * 7, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )
            
        self.head = nn.Sequential(
            nn.Conv2d(256 * 4, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fpn_backbone is None:
            x_norm = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
            return F.normalize(self.fallback_head(x_norm), p=2, dim=1)
            
        # Convert [-1, 1] input image tensor to torchvision [0, 1] format
        x_norm = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
        
        # Extract intermediate FPN feature representations before output layer
        features = self.fpn_backbone(x_norm)
        h, w = features['0'].shape[2], features['0'].shape[3]
        
        p2 = features['0']
        p3 = F.interpolate(features['1'], size=(h, w), mode='bilinear', align_corners=False)
        p4 = F.interpolate(features['2'], size=(h, w), mode='bilinear', align_corners=False)
        p5 = F.interpolate(features['3'], size=(h, w), mode='bilinear', align_corners=False)
        
        fused = torch.cat([p2, p3, p4, p5], dim=1) # Fused multi-scale representation (B, 1024, H, W)
        emb = self.head(fused)
        return F.normalize(emb, p=2, dim=1)

def fasterrcnn_backbone(embedding_dim=512, **kwargs):
    return FasterRCNNBackbone(embedding_dim=embedding_dim)

# Default Alias
ResNet100Backbone = iresnet100

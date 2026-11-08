import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class PositionEmbeddingRandom(nn.Module):
    """
    Official Positional Encoding using Random Spatial Frequencies Matrix.
    Reference: Meta AI Segment Anything (SAM) Implementation (Kirillov et al., ICCV 2023).
    """
    def __init__(self, num_pos_feats: int = 128, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix", torch.randn((2, num_pos_feats)) * scale)

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1).unsqueeze(0)

class SAMAttention(nn.Module):
    """
    Official Multi-Head Attention Block for SAM Transformer Mask Decoder (Meta AI SAM Reference).
    """
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.head_dim = self.internal_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b_q, n_q, _ = q.shape
        b_k, n_k, _ = k.shape

        q = self.q_proj(q).view(b_q, n_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(b_k, n_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(b_k, n_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(b_q, n_q, self.internal_dim)
        return self.out_proj(out)

class TwoWayAttentionBlock(nn.Module):
    """
    Official SAM Two-Way Attention Block (Meta AI SAM Reference).
    Performs self-attention and bi-directional cross-attention between prompts and image features.
    """
    def __init__(self, embedding_dim: int, num_heads: int, mlp_dim: int = 2048) -> None:
        super().__init__()
        self.self_attn = SAMAttention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_img = SAMAttention(embedding_dim, num_heads)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim)
        )
        self.norm3 = nn.LayerNorm(embedding_dim)
        
        self.cross_attn_img_to_token = SAMAttention(embedding_dim, num_heads)
        self.norm4 = nn.LayerNorm(embedding_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, query_pe: torch.Tensor, key_pe: torch.Tensor):
        # 1. Self attention on prompt tokens
        q = queries + query_pe
        attn = self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries + attn)

        # 2. Cross attention: prompt tokens to image feature embeddings
        q = queries + query_pe
        k = keys + key_pe
        attn = self.cross_attn_token_to_img(q=q, k=k, v=keys)
        queries = self.norm2(queries + attn)

        # 3. LayerNorm MLP
        mlp_out = self.mlp(queries)
        queries = self.norm3(queries + mlp_out)

        # 4. Cross attention: image feature embeddings to prompt tokens
        q = keys + key_pe
        k = queries + query_pe
        attn = self.cross_attn_img_to_token(q=q, k=k, v=queries)
        keys = self.norm4(keys + attn)

        return queries, keys

class TwoWayTransformer(nn.Module):
    """
    Official SAM Two-Way Transformer Decoder (Meta AI SAM Reference).
    """
    def __init__(self, depth: int = 2, embedding_dim: int = 256, num_heads: int = 8, mlp_dim: int = 2048):
        super().__init__()
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_dim=mlp_dim)
            for _ in range(depth)
        ])
        self.final_attn_token_to_img = SAMAttention(embedding_dim, num_heads)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: torch.Tensor, image_pe: torch.Tensor, point_embedding: torch.Tensor):
        b, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1) # (B, HW, C)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)

        q = queries + point_embedding
        k = keys + image_pe
        attn = self.final_attn_token_to_img(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attn)

        return queries, keys.permute(0, 2, 1).view(b, c, h, w)

class SAMMaskDecoder(nn.Module):
    """
    Official SAM Mask Decoder & IoU Score Predictor (Meta AI SAM Reference).
    Predicts multi-mask foreground segmentations and IoU prediction scores.
    """
    def __init__(self, transformer_dim: int = 256, num_multimask_outputs: int = 3):
        super().__init__()
        self.transformer = TwoWayTransformer(depth=2, embedding_dim=transformer_dim, num_heads=8, mlp_dim=2048)
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            nn.GroupNorm(1, transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.GELU(),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(num_multimask_outputs + 1)
        ])

        self.iou_prediction_head = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_multimask_outputs + 1)
        )

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor, sparse_prompt_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # (5, C)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        
        tokens = torch.cat([output_tokens, sparse_prompt_embeddings], dim=1)
        hs, src = self.transformer(image_embeddings, image_pe, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_multimask_outputs + 1), :]

        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_multimask_outputs + 1):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1) # (B, 4, C_sub)

        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

class SAMFaceSegmentor(nn.Module):
    """
    Lightweight prompt-compatible foreground mask module.

    This is not a drop-in Meta SAM checkpoint or full SAM image encoder. It is
    retained as an optional learned preprocessor and must be checkpointed
    before it is enabled in recognition.
    """
    def __init__(self, embed_dim: int = 256, image_size: int = 112):
        super(SAMFaceSegmentor, self).__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim

        # Official ViT Image Encoder Stem with Patch Embedding (Patch Size 16)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=8, padding=4)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Official Random Gaussian Positional Encoding Matrix (Meta AI SAM Reference)
        self.prompt_pe = PositionEmbeddingRandom(num_pos_feats=embed_dim // 2)

        # Official Two-Way Transformer Mask Decoder & IoU Head (Meta AI SAM Reference)
        self.mask_decoder = SAMMaskDecoder(transformer_dim=embed_dim, num_multimask_outputs=3)

        # Prompt Encoders for Bounding Boxes & Points
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(4)
        ])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

    def forward(self, img_crop: torch.Tensor, box_prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            img_crop: (B, 3, H, W) normalized input face crop [-1, 1]
            box_prompt: (B, 4) optional bounding box prompt coordinates [x1, y1, x2, y2]
        Returns:
            mask_sam: (B, 1, H, W) SAM face foreground segmentation mask in [0, 1]
        """
        B, C, H, W = img_crop.shape

        # Step 1: ViT Patch Feature Extraction
        feat = self.patch_embed(img_crop) # (B, 256, 14, 14)
        b, c, h_f, w_f = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1) # (B, 14, 14, 256)
        feat_norm = self.encoder_norm(feat_flat).permute(0, 3, 1, 2)

        # Step 2: Generate Gaussian Positional Encoding
        image_pe = self.prompt_pe((h_f, w_f)).to(img_crop.device)
        image_pe = image_pe.expand(B, -1, -1, -1)

        # Step 3: Sparse Prompt Embedding (Points / Boxes)
        sparse_prompts = self.not_a_point_embed.weight.unsqueeze(0).expand(B, 2, -1) # (B, 2, C)

        # Step 4: Two-Way Transformer Mask Decoder
        masks, iou_pred = self.mask_decoder(
            image_embeddings=feat_norm,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompts
        )

        # Select primary face mask and upsample to original crop size (H, W)
        primary_mask = torch.sigmoid(masks[:, 0:1, :, :]) # (B, 1, 28, 28)
        if box_prompt is not None:
            if box_prompt.shape != (B, 4):
                raise ValueError(f"box_prompt must have shape ({B},4)")
            _, _, mask_h, mask_w = primary_mask.shape
            prior = torch.zeros_like(primary_mask)
            for index, box in enumerate(box_prompt):
                x1, y1, x2, y2 = box
                ix1 = int(torch.floor(x1 / max(float(W), 1.0) * mask_w).clamp(0, mask_w - 1).item())
                iy1 = int(torch.floor(y1 / max(float(H), 1.0) * mask_h).clamp(0, mask_h - 1).item())
                ix2 = int(torch.ceil(x2 / max(float(W), 1.0) * mask_w).clamp(ix1 + 1, mask_w).item())
                iy2 = int(torch.ceil(y2 / max(float(H), 1.0) * mask_h).clamp(iy1 + 1, mask_h).item())
                prior[index, :, iy1:iy2, ix1:ix2] = 1.0
            primary_mask = primary_mask * prior
        face_mask_sam = F.interpolate(primary_mask, size=(H, W), mode='bilinear', align_corners=False).clamp(0.0, 1.0)

        return face_mask_sam

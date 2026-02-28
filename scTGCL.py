import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import mask_attention_matrix


# ============================================================================
# Contrastive Loss
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Contrastive loss between original and augmented representations"""
    
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_orig, z_aug):
        """
        Args:
            z_orig: Original embeddings [batch_size, embed_dim]
            z_aug: Augmented embeddings [batch_size, embed_dim]
            
        Returns:
            loss: Contrastive loss value
        """
        batch_size = z_orig.shape[0]
        
        # Normalize embeddings
        z_orig = F.normalize(z_orig, dim=1)
        z_aug = F.normalize(z_aug, dim=1)
        
        # Compute similarity matrices
        sim_orig_aug = torch.mm(z_orig, z_aug.T) / self.temperature
        sim_orig_orig = torch.mm(z_orig, z_orig.T) / self.temperature
        sim_aug_aug = torch.mm(z_aug, z_aug.T) / self.temperature
        
        # Positive pairs: (i, i) - same cell in original and augmented
        positive_sim = torch.diag(sim_orig_aug)
        
        # Compute loss for original -> augmented
        exp_sim_orig_aug = torch.exp(sim_orig_aug)
        exp_sim_orig_orig = torch.exp(sim_orig_orig)
        
        # Remove diagonal from negative samples
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_orig.device)
        exp_sim_orig_orig = exp_sim_orig_orig.masked_fill(mask, 0)
        
        denominator = exp_sim_orig_aug.sum(dim=1) + exp_sim_orig_orig.sum(dim=1)
        loss_orig_to_aug = -torch.log(torch.exp(positive_sim) / (denominator + 1e-8))
        
        # Compute loss for augmented -> original (symmetric)
        exp_sim_aug_orig = torch.exp(sim_orig_aug.T)
        exp_sim_aug_aug = torch.exp(sim_aug_aug)
        exp_sim_aug_aug = exp_sim_aug_aug.masked_fill(mask, 0)
        
        denominator_aug = exp_sim_aug_orig.sum(dim=1) + exp_sim_aug_aug.sum(dim=1)
        loss_aug_to_orig = -torch.log(torch.exp(positive_sim) / (denominator_aug + 1e-8))
        
        # Total contrastive loss
        loss = (loss_orig_to_aug.mean() + loss_aug_to_orig.mean()) / 2
        
        return loss


# ============================================================================
# Multi-Head Attention
# ============================================================================

class EnhancedMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with support for attention masking"""
    
    def __init__(self, embed_dim=512, num_heads=8):
        super(EnhancedMultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Weight matrices for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.scale = np.sqrt(self.head_dim)
        
    def forward(self, x, mask_attention=False, attention_mask_prob=0.2):
        """
        Args:
            x: [batch_size, embed_dim]
            mask_attention: Whether to apply attention masking
            attention_mask_prob: Probability of masking attention weights
            
        Returns:
            output: [batch_size, embed_dim]
            attention_weights: [num_heads, batch_size, batch_size]
            qkv_matrices: Dictionary containing Q, K, V matrices
        """
        batch_size = x.shape[0]
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply masking for augmented data
        if mask_attention:
            attention_weights = mask_attention_matrix(attention_weights, attention_mask_prob)
        
        # Apply attention to values
        attention_output = torch.bmm(attention_weights, V)
        attention_output = attention_output.transpose(0, 1).contiguous()
        attention_output = attention_output.view(batch_size, self.embed_dim)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        # Store Q, K, V for analysis
        qkv_matrices = {
            'Q': Q.transpose(0, 1),
            'K': K.transpose(0, 1),
            'V': V.transpose(0, 1),
        }
        
        return output, attention_weights, qkv_matrices


# ============================================================================
# Main Model
# ============================================================================

class ContrastiveTransformerAutoencoder(nn.Module):
    """Transformer-based Autoencoder with Contrastive Learning"""
    
    def __init__(self, input_dim, embed_dim=512, num_heads=8, latent_dim=128, 
                 dropout=0.1, mask_prob=0.15, attention_mask_prob=0.2):
        super(ContrastiveTransformerAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.mask_prob = mask_prob
        self.attention_mask_prob = attention_mask_prob
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2048),
            nn.Linear(2048, embed_dim),
        )
        
        # Multi-head attention
        self.multihead_attention = EnhancedMultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Encoder output (latent space)
        self.encoder_output = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, embed_dim // 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.Linear(embed_dim, 2048),  
            nn.Linear(2048, input_dim)
        )
        
    def forward(self, x, x_aug=None, is_training=True):
        """
        Args:
            x: Original input [batch_size, input_dim]
            x_aug: Augmented input [batch_size, input_dim] (optional)
            is_training: Whether in training mode
            
        Returns:
            Dictionary containing all outputs
        """
        # Process original data
        embedded = self.input_projection(x)
        attn_output, attn_weights, qkv_orig = self.multihead_attention(
            embedded, mask_attention=False
        )
        
        # Residual + LayerNorm
        x1 = self.ln1(embedded + attn_output)
        
        # Latent representation
        latent = self.encoder_output(x1)
        
        # Projection for contrastive learning
        z_orig = self.projection_head(latent)
        
        # Reconstruction
        reconstructed = self.decoder(latent)
        
        outputs = {
            'reconstructed': reconstructed,
            'latent': latent,
            'z_orig': z_orig,
            'x2_orig': x1,
            'attention_weights_orig': attn_weights,
            'qkv_orig': qkv_orig
        }
        
        # Process augmented data if provided
        if x_aug is not None and is_training:
            embedded_aug = self.input_projection(x_aug)
            attn_output_aug, attn_weights_aug, qkv_aug = self.multihead_attention(
                embedded_aug,
                mask_attention=True,
                attention_mask_prob=self.attention_mask_prob
            )
            
            x1_aug = self.ln1(embedded_aug + attn_output_aug)
            latent_aug = self.encoder_output(x1_aug)
            z_aug = self.projection_head(latent_aug)
            reconstructed_aug = self.decoder(latent_aug)
            
            outputs.update({
                'reconstructed_aug': reconstructed_aug,
                'latent_aug': latent_aug,
                'z_aug': z_aug,
                'x2_aug': x1_aug,
                'attention_weights_aug': attn_weights_aug,
                'qkv_aug': qkv_aug
            })
        
        return outputs
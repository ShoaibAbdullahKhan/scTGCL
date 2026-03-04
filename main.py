import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import scanpy as sc
from config import get_config
from utils import (
    setup_seed, 
    create_augmented_data, 
    scRNASeqDataset, 
    evaluate
)
from scTGCL import ContrastiveLoss, scTGCL


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    """Parse command line arguments, using config defaults"""
    defaults = get_config()

    parser = argparse.ArgumentParser(
        description='scTGCL: Single-cell Transformer Graph Contrastive Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument('--dataset', type=str, help='Dataset name to use for training')
    parser.add_argument("--n_clusters", type=int, help='Number of classes in the dataset')
    
    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=defaults['embed_dim'],
                        help='Embedding dimension for transformer')
    parser.add_argument('--num_heads', type=int, default=defaults['num_heads'],
                        help='Number of attention heads')
    parser.add_argument('--latent_dim', type=int, default=defaults['latent_dim'],
                        help='Latent space dimension')
    parser.add_argument('--dropout', type=float, default=defaults['dropout'],
                        help='Dropout rate')

    # Data augmentation
    parser.add_argument('--mask_prob', type=float, default=defaults['mask_prob'],
                        help='Gene masking probability for augmentation')
    parser.add_argument('--attention_mask_prob', type=float, default=defaults['attention_mask_prob'],
                        help='Attention masking probability')

    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=defaults['lambda_recon'],
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_impute', type=float, default=defaults['lambda_impute'],
                        help='Weight for imputation loss')
    parser.add_argument('--lambda_contrast', type=float, default=defaults['lambda_contrast'],
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=defaults['temperature'],
                        help='Temperature for contrastive loss')

    # Training
    parser.add_argument('--lr', type=float, default=defaults['lr'],
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay'],
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=defaults['seed'],
                        help='Random seed (default is randomly chosen at startup)')

    # Paths
    parser.add_argument('--save_dir', type=str, default=defaults['save_dir'],
                        help='Directory to save results and visualizations')

    return parser.parse_args()


# ============================================================================
# Training Function
# ============================================================================

def train_model(model, train_loader, test_loader, config, device):
    """Train the contrastive transformer autoencoder"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    contrastive_criterion = ContrastiveLoss(temperature=config['temperature'])
    
    # Loss weights
    lambda_recon = config['lambda_recon']
    lambda_impute = config['lambda_impute']
    lambda_contrast = config['lambda_contrast']
    
    best_loss = float('inf')
    best_latent = None
    best_labels = None
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    print(f"Loss Weights: Recon={lambda_recon}, Impute={lambda_impute}, Contrast={lambda_contrast}")
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_impute_loss = 0
        train_contrast_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            
            # Create augmented data
            batch_x_aug, gene_mask = create_augmented_data(
                batch_x, 
                mask_prob=config['mask_prob']
            )
            
            # Forward pass
            outputs = model(batch_x, batch_x_aug, is_training=True)
            
            # Compute losses
            loss_recon = mse_criterion(outputs['reconstructed'], batch_x)
            loss_impute = mse_criterion(outputs['reconstructed_aug'], batch_x)
            loss_contrast = contrastive_criterion(outputs['z_orig'], outputs['z_aug'])
            
            # Total loss
            loss = (lambda_recon * loss_recon + 
                   lambda_impute * loss_impute + 
                   lambda_contrast * loss_contrast)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_recon.item()
            train_impute_loss += loss_impute.item()
            train_contrast_loss += loss_contrast.item()
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_loss = 0
        all_latent = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x, x_aug=None, is_training=False)
                
                loss = mse_criterion(outputs['reconstructed'], batch_x)
                test_loss += loss.item()
                all_latent.append(outputs['latent'].cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())
        
        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            best_latent = np.vstack(all_latent)
            best_labels = np.hstack(all_labels)
                
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Total={train_loss:.4f} "
                  f"(Recon={train_recon_loss:.4f}, "
                  f"Impute={train_impute_loss:.4f}, "
                  f"Contrast={train_contrast_loss:.4f}), "
                  f"Test={test_loss:.4f}, Time={elapsed:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    print("="*70)
    
    return best_latent, best_labels, total_time


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution pipeline driven by command line arguments"""

    args = parse_args()

    # Build config from parsed args (all defaults already come from get_config())
    config = vars(args).copy()
    data_path = config.pop('dataset')

    print(f"\n{'#'*70}")
    print(f"Running scTGCL on '{data_path}' dataset")
    print(f"{'#'*70}")
    print("\nActive Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\nRandom Seed: {config['seed']}")
    setup_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading dataset: {data_path}")
    data = sc.read_h5ad(data_path)
    X_all = data.X
    y_all = data.obs['cell_type'].values
    
    print(f"Data shape: {X_all.shape}, Labels shape: {y_all.shape}")
    print(f"Cell type distribution:\n{pd.DataFrame(y_all).value_counts()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(pd.DataFrame(y_all).values.ravel())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=config['seed'], stratify=y_all
    )
    
    # Create datasets and loaders
    train_dataset = scRNASeqDataset(X_train, y_train)
    test_dataset = scRNASeqDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = scTGCL(
        input_dim=X_all.shape[1],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
        mask_prob=config['mask_prob'],
        attention_mask_prob=config['attention_mask_prob']
    ).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    best_latent, best_labels, training_time = train_model(
        model, train_loader, test_loader, config, device
    )
    
    # Evaluate
    acc, nmi, ari = evaluate(
        latent=best_latent, 
        labels=best_labels, 
        n_clusters=config['n_clusters'],
        seed=config['seed'],
        save_dir=config['save_dir']
    )
    
    
    return model, acc, nmi, ari


if __name__ == "__main__":
    main()
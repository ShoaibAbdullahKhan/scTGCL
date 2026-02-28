def get_config():
    """
    Get default configuration for scTGCL model
    
    Returns:
        dict: Configuration dictionary with all hyperparameters
    """
    config = {
        # Model architecture
        'embed_dim': 512,
        'num_heads': 8,
        'latent_dim': 64,
        'dropout': 0.3,
        
        # Data augmentation
        'mask_prob': 0.25,              # Gene masking probability
        'attention_mask_prob': 0.35,    # Attention masking probability
        
        # Loss weights
        'lambda_recon': 6,              # Reconstruction loss weight
        'lambda_impute': 1,             # Imputation loss weight
        'lambda_contrast': 0.8,         # Contrastive loss weight
        'temperature': 0.5,             # Contrastive loss temperature
        
        # Training
        'lr': 0.001,
        'weight_decay': 1e-3,
        'epochs': 100,
        'batch_size': 128,
        'seed': 35,
        
        # Paths
        'save_dir': 'results',
    }
    
    return config

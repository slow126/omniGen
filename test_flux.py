def test_flux_vae(dataset_dir):
    # Load the VAE model
    model = VAE.load_from_checkpoint("flux_vae/diffusion_pytorch_model.safetensors")
    
    # Load the dataset
    dataset = CustomDataset(dataset_dir)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Initialize the PyTorch Lightning trainer
    trainer = pl.Trainer()
    
    # Test the model
    trainer.test(model, dataloaders=dataloader)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Flux VAE on a dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory.")
    args = parser.parse_args()
    
    test_flux_vae(args.dataset_dir)

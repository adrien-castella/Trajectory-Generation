import torch
from dataset_loader import ArgoverseDataset
from visualization_tools import forward_and_reverse_diffusion, generate_and_plot_diffusion
from models import CustomNoiseScheduler, TabDDPM_Transformer

# Your classes should already be defined elsewhere:
# from your_module import ArgoverseDataset, CustomNoiseScheduler, YourModelClass

# Example usage
if __name__ == "__main__":
    # --- Configuration ---
    model_path = "argoverse_v1.2/train_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_agents = 2
    t_for_forward_diffusion = 200

    # --- Load Dataset ---
    dataset = ArgoverseDataset(split='train', num_agents=n_agents)  # updated per your request

    # --- Load Model ---
    model = TabDDPM_Transformer(
        input_dim=4,
        seq_len=40,
        hidden_dim=128,
        embed_dim=128,
        num_layers=4,
        dropout=0.25
    )  # replace with actual model class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # --- Create Noise Scheduler ---
    noise_scheduler = CustomNoiseScheduler(num_indices=[0,1,2,3])

    # --- Run Diffusion from Noise ---
    # generate_and_plot_diffusion(
    #     model=model,
    #     noise_scheduler=noise_scheduler,
    #     dataset=dataset,
    #     n=n_agents,
    #     save_dir="generate_from_noise",
    #     device=device
    # )
    
    noise_scheduler.reset_scheduler()

    # --- Run Forward + Reverse Diffusion ---
    forward_and_reverse_diffusion(
        model=model,
        noise_scheduler=noise_scheduler,
        dataset=dataset,
        t=t_for_forward_diffusion,
        save_dir="denoise_train_sample",
        device=device
    )
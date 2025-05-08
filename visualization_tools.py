import random, torch, os
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectories(tensor1, tensor2, save_dir="plots", set_axes=False):
    """
    Plots a comparison of two tensors of shape (N, H, 4).
    Each of the N plots contains two subplots: one for tensor1[n] and one for tensor2[n].

    Arguments:
    - tensor1, tensor2: torch tensors of shape (N, H, 4)
    - set_axes: if True, match x/y/color axes across both subplots in each figure
    - save_dir: directory where images will be saved
    """
    assert tensor1.shape == tensor2.shape, "tensor1 and tensor2 must have the same shape"
    assert tensor1.shape[2] == 4, "Last dimension must be of size 4 (x, y, color, binary)"

    os.makedirs(save_dir, exist_ok=True)

    N, H, _ = tensor1.shape

    for n in range(N):
        t1 = tensor1[n].detach().cpu().numpy()
        t2 = tensor2[n].detach().cpu().numpy()

        mask1 = t1[:, 3] <= 0.5
        mask2 = t2[:, 3] <= 0.5

        fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)

        if set_axes:
            all_x = np.concatenate([t1[mask1, 0], t2[mask2, 0]])
            all_y = np.concatenate([t1[mask1, 1], t2[mask2, 1]])
            all_c = np.concatenate([t1[mask1, 2], t2[mask2, 2]])
            xlim = (all_x.min(), all_x.max())
            ylim = (all_y.min(), all_y.max())
            clim = (all_c.min(), all_c.max())
        else:
            xlim = ylim = clim = (None, None)

        for data, mask, ax, title in zip(
            [t1, t2],
            [mask1, mask2],
            axs,
            ["Tensor 1", "Tensor 2"]
        ):
            x = data[mask, 0]
            y = data[mask, 1]
            c = data[mask, 2]
            indices = np.arange(H)[mask]

            sc = ax.scatter(x, y, c=c, cmap="viridis", edgecolor="k",
                            vmin=clim[0], vmax=clim[1])

            # Add black labels *next to* each point
            for xi, yi, idx in zip(x, y, indices):
                ax.text(xi, yi, str(idx), fontsize=8, color="black")

            ax.set_title(f"{title} - Agent {n}")
            ax.set_xlabel("X (dim 0)")
            ax.set_ylabel("Y (dim 1)")

            if set_axes:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)

        plt.subplots_adjust(wspace=0.3)
        plt.savefig(os.path.join(save_dir, f"agent_{n}.png"))
        plt.close(fig)

def denormalize(tensor, dataset):
    """
    De-normalizes a tensor using the mean and std from the dataset.
    Assumes tensor shape is (B, N*H, W) and mean/std shape is (B, N, H, W)
    """
    # Reshape tensor from (B, N*H, W) to (B, N, H, W)
    B, _, _ = tensor.shape
    N, H, W = dataset.mean.shape  # Get N from the mean tensor
    tensor_reshaped = tensor.view(B, N, H, W).cpu()
    
    # Now tensor_reshaped and mean/std have matching dimensions
    denormalized = tensor_reshaped * dataset.std + dataset.mean
    # Return in original shape
    return denormalized.squeeze(0)


def generate_and_plot_diffusion(model, noise_scheduler, dataset, n=1, save_dir='diffusion_progress', device='cpu'):
    """
    Generates `n` samples from noise and denoises them, visualizing the process.
    Plots the original noise next to denoised outputs every 100 steps.

    Parameters:
        model: Noise prediction model
        noise_scheduler: Object with reverse_diffusion(x, noise, t)
        dataset: Used to access dataset.mean and dataset.std for denormalization
        n (int): Number of agents
        max_timestep (int): Max diffusion step
        save_dir (str): Output folder
        device (str): Device
    """
    os.makedirs(save_dir, exist_ok=True)

    x = torch.randn(1, n * 20, 4, device=device)
    original_noise = x.clone()
    max_timestep = noise_scheduler.num_infer_timesteps

    for t in range(max_timestep):
        current_t = noise_scheduler.get_reverse_timestep(t)
        t_tensor = torch.full((1,), current_t, dtype=torch.long, device=device)
        predicted_noise = model(x, t_tensor)
        x = noise_scheduler.diffusion_generate(x, predicted_noise, current_t)

        if t % 10 == 0 or t == max_timestep - 1:
            step_idx = max_timestep - t
            step_folder = os.path.join(save_dir, f'step_{step_idx:04d}')
            os.makedirs(step_folder, exist_ok=True)

            plot_trajectories(
                denormalize(original_noise, dataset),
                denormalize(x.detach(), dataset),
                save_dir=step_folder
            )
    
    step_folder = os.path.join(save_dir, f'final_output')
    plot_trajectories(
        denormalize(original_noise, dataset),
        denormalize(x.detach(), dataset),
        save_dir=step_folder
    )

    return x.detach()


def forward_and_reverse_diffusion(model, noise_scheduler, dataset, t=1000, save_dir='sample_reconstruction', device='cpu'):
    """
    Samples `n` real points from dataset, adds noise at timestep t, and denoises back.
    Visualizes original vs denoised every 100 steps.

    Parameters:
        model: Noise prediction model
        noise_scheduler: Object with forward_diffusion(x, t) and reverse_diffusion(x, noise, t)
        dataset: Dataset instance with mean and std
        t (int): Diffusion step to start from
        save_dir (str): Output folder
        device (str): Device
    """
    os.makedirs(save_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), 1)
    x_start = torch.stack([dataset[i]['input'].clone() for i in indices]).to(device)  # shape (n, 50, 3)
    B, N, H, W = x_start.shape
    x_start = x_start.view(B, N*H, W)  # Reshape to (1, N*H, W)
    x, _, _ = noise_scheduler.forward_diffusion(x_start, torch.tensor(t))

    max_timestep = t

    for t in range(max_timestep, 0, -1):
        # current_t = noise_scheduler.get_reverse_timestep(t)
        t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
        predicted_noise = model(x, t_tensor)
        x = noise_scheduler.reverse_diffusion(x, predicted_noise, t_tensor)

        if t % 100 == 0 or t == 1:
            step_idx = t
            step_folder = os.path.join(save_dir, f'step_{step_idx:04d}')
            os.makedirs(step_folder, exist_ok=True)

            plot_trajectories(
                denormalize(x_start, dataset).cpu(),
                denormalize(x.detach(), dataset).cpu(),
                save_dir=step_folder,
                set_axes=False
            )
    
    step_folder = os.path.join(save_dir, f'final_output')
    plot_trajectories(
        denormalize(x_start, dataset).cpu(),
        denormalize(x.detach(), dataset).cpu(),
        save_dir=step_folder,
        set_axes=True
    )

    return x.detach()
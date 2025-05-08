import torch, os, warnings, argparse, json
import torch.optim as optim
from models import CustomNoiseScheduler, TabDDPM_Transformer
from load_data import load_super
from training_utils import train_epoch, test_model, EMA
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model with configurable hyperparameters.")

    parser.add_argument('--num_agents', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=2500)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--input_dim', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=20)
    parser.add_argument('--save_folder', type=str, default="argoverse_v1.1")
    parser.add_argument('--dataset', type=str, default="Argoverse", choices=['Argoverse', 'Trajectory', 'Road'])
    parser.add_argument('--testing_split', type=float, default=0.2)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--numerical_indices', type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--categorical_indices', type=str, default="[]", help='Use JSON format, e.g. [[5,6,7],[8,9]]')
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate for AdamW optimizer")
    parser.add_argument('--eta_min', type=float, default=1e-7, help="Minimum learning rate for CosineAnnealingLR scheduler")
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate for the model')
    parser.add_argument('--layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--width', type=int, default=256, help='The width of the transformer network')
    parser.add_argument('--embed_dim', type=int, default=256, help='The dimension of the embeddings')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='The decay rate for EMA')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for the AdamW optimizer')
    parser.add_argument('--use_importance', action='store_true', help='Enable importance weighting, putting more weight on smaller noise steps')

    args = parser.parse_args()
    
    # Parse the JSON string into actual list of lists
    try:
        args.categorical_indices = json.loads(args.categorical_indices)
        assert all(isinstance(group, list) for group in args.categorical_indices)
    except Exception as e:
        raise ValueError(f"Invalid format for categorical_indices. Must be a JSON list of lists. Got: {args.categorical_indices}") from e

    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.save_folder, exist_ok=True)

    # ==== Model, Data, Scheduler ====
    model = TabDDPM_Transformer(
        input_dim=args.input_dim, 
        seq_len=args.seq_len * args.num_agents, 
        hidden_dim=args.width, 
        embed_dim=args.embed_dim,
        num_layers=args.layers, 
        dropout=args.dropout
    ).to(device)
    ema_model = EMA(model, args.ema_decay)
    ema_model.to(device)

    noise_scheduler = CustomNoiseScheduler(
        num_train_timesteps=args.num_timesteps,
        num_indices=args.numerical_indices,
        cat_indices=[args.categorical_indices[i:i+3] for i in range(0, len(args.categorical_indices), 3)]
    )

    # Determine inputs based on dataset
    if args.dataset == 'Argoverse':
        inputs = [args.batch_size, args.num_agents]
    else:
        inputs = [args.batch_size, args.seq_len, args.testing_split, args.validation_split]

    train_loader, val_loader, test_loader = load_super(inputs, args.dataset)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.eta_min)

    start_epoch = 0
    best_val_loss = float('inf')
    model_save_path = os.path.join(args.save_folder, 'train_model.pt')

    for epoch in range(start_epoch, args.num_epochs):
        best_val_loss = train_epoch(
            model, ema_model, train_loader, val_loader, optimizer,
            scheduler, noise_scheduler, device, args.numerical_indices,
            model_save_path, epoch, args.num_epochs, best_val_loss
        )

    test_loss, ema_loss = test_model(
        model, ema_model, test_loader, noise_scheduler,
        device, args.numerical_indices, model_save_path
    )

    torch.save({'test_loss': [test_loss, ema_loss], 'best_val_loss': best_val_loss}, os.path.join(args.save_folder, 'losses.pt'))


if __name__ == "__main__":
    main()
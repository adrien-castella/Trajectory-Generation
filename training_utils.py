import torch
import torch.nn.functional as F
from tqdm import tqdm

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.ema_model = self.clone_model(model)
        self.decay = decay
        self.step = 0
    
    def __call__(self, input, t):
        return self.ema_model(input, t)

    def clone_model(self, model):
        """Creates a copy of the model."""
        model_copy = model.__class__(*model.inputs)
        model_copy.load_state_dict(model.state_dict())
        return model_copy

    def update(self):
        """Update EMA model parameters with exponential decay."""
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data = ema_param.data * self.decay + param.data * (1.0 - self.decay)
        self.step += 1
    
    def train(self):
        self.ema_model.train()
    
    def eval(self):
        self.ema_model.eval()
    
    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
    
    def to(self, device):
        self.ema_model.to(device)

    def get_model(self):
        """Return the EMA model."""
        return self.ema_model

def train_batch(batch, model, noise_scheduler, device, numerical_indices):
    clean_seq = batch['input'].clone().to(device)

    B, H, T, W = clean_seq.shape
    clean_seq = clean_seq.view(B, H*T, W)

    noisy_seq, num_noise, timesteps = noise_scheduler.forward_diffusion(clean_seq)
    pred_noise = model(noisy_seq, timesteps)
    noise_loss = F.mse_loss(pred_noise[:, :, numerical_indices], num_noise)
    
    return noise_loss

def train_epoch(model, ema_model, train_loader, val_loader, optimizer, scheduler, noise_scheduler, 
                device, numerical_indices, model_save_path, epoch, num_epochs, best_val_loss, ema_decay=0.999):
    model.train()
    train_loss = 0.0
    
    for batch in tqdm(train_loader, desc=f"Training epoch {epoch+1}", leave=False):
        loss = train_batch(batch, model, noise_scheduler, device, numerical_indices)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Update EMA model parameters
        ema_model.update()
        
        train_loss += loss.item()

    scheduler.step()
    train_loss /= len(train_loader)
    
    # Validation for Training Model
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating train model {epoch+1}", leave=False):
            loss = train_batch(batch, model, noise_scheduler, device, numerical_indices)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Validation for EMA Model
    ema_model.eval()
    ema_val_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating EMA model {epoch+1}", leave=False):
            loss = train_batch(batch, ema_model, noise_scheduler, device, numerical_indices)
            ema_val_loss += loss.item()
    
    ema_val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss (Training Model): {val_loss:.6f} | Val Loss (EMA Model): {ema_val_loss:.6f}")
    
    # Save best models
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Saved new best model (Training Model) at epoch {epoch+1}!")
    
    # Save EMA model
    if ema_val_loss < best_val_loss:
        best_val_loss = ema_val_loss
        torch.save(ema_model.get_model().state_dict(), model_save_path.replace("train", "ema"))
        print(f"💾 Saved new best EMA model at epoch {epoch+1}!")
    
    return best_val_loss

def test_model(model, ema_model, test_loader, noise_scheduler, device, numerical_indices, model_save_path):
    print("🧪 Testing best training model...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            num_loss = train_batch(batch, model, noise_scheduler, device, numerical_indices)
            test_loss += num_loss.item()

    test_loss /= len(test_loader)
    print(f"🎯 Final Test Loss (Training Model): {test_loss:.6f}")
    
    # Test EMA model
    print("🧪 Testing best EMA model...")
    ema_model.load_state_dict(torch.load(model_save_path.replace("train", "ema")))
    ema_model.eval()
    ema_test_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            num_loss = train_batch(batch, ema_model, noise_scheduler, device, numerical_indices)
            ema_test_loss += num_loss.item()

    ema_test_loss /= len(test_loader)
    print(f"🎯 Final Test Loss (EMA Model): {ema_test_loss:.6f}")

    return test_loss, ema_test_loss
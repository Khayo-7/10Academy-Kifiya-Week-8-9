import torch
from models.model import FraudDetectionModel  # Your trained model

def optimize_model():
    """Convert the PyTorch model to TorchScript for faster inference."""
    model = FraudDetectionModel()
    model.load_state_dict(torch.load("models/fraud_model.pth"))
    model.eval()
    
    # Convert to TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save("models/fraud_model_scripted.pt")
    
    print("✅ Model optimized and saved as TorchScript.")

if __name__ == "__main__":
    optimize_model()


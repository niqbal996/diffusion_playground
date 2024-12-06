
import torch 
import torch.nn.functional as F

def calculate_output_difference(output1, output2):
    """Calculate various difference metrics between two outputs"""
    with torch.no_grad():
            l1_diff = torch.abs(output1 - output2).mean().item()
            l2_diff = torch.sqrt(((output1 - output2) ** 2).mean()).item()
            cosine_sim = (
                F.cosine_similarity(output1.flatten(1), output2.flatten(1))
            .mean()
            .item() 
        )
    return l1_diff, l2_diff, cosine_sim

def check_tensor(tensor, name=""):
    """Debug function to check tensor values"""
    if torch.isnan(tensor).any():
        accelerator.print(f"NaN detected in {name}")
        return False
    if torch.isinf(tensor).any():
        accelerator.print(f"Inf detected in {name}")
        return False
    return True
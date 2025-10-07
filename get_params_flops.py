import json
from thop import profile  # Need to install thop library: pip install thop
import torch
import gc
from pathlib import Path

from TFNet import TFNet, reparameterize_model, create_tfnet_variants


def calculate_model_stats(model, model_name, input_shape=(2, 3, 512, 512), output_dir="./params"):
    """
    Calculate model parameters and FLOPs, and save as JSON file.

    Args:
        model: Model to be calculated
        model_name: Name of the model
        input_shape: Input tensor shape, defaults to (2, 3, 512, 512)
        output_dir: Output directory, defaults to "./params"

    Returns:
        dict: Dictionary containing model information
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    input_tensor = torch.randn(*input_shape).to(device)
    batch_size = input_shape[0]

    model = model.to(device)
    model.eval()

    parameters = sum(p.numel() for p in model.parameters())
    parameters_M = parameters / 1e6  # Convert to millions (M)

    flops, _ = profile(model, inputs=(input_tensor,))
    flops = flops / batch_size  # Normalize by batch size
    flops_G = flops / 1e9  # Convert to billions (G)

    model_info = {
        "model_name": model_name,
        "parameters(M)": round(parameters_M, 2),  # Keep 2 decimal places
        "flops(G)": round(flops_G, 2),  # Keep 2 decimal places
        "input_shape": input_shape
    }

    json_filename = f"{output_dir}/{model_name}.json"
    with open(json_filename, 'w') as f:
        json.dump(model_info, f, indent=4)

    print(f"{model_name}, "
          f"Parameters: {parameters_M:.3f}M, FLOPs: {flops_G:.3f}G, "
          f"Input shape: {input_shape}")

    return model_info


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    imgz = 512
    input_shape = (batch_size, 3, imgz, imgz)
    nc = 2  # Number of classes
    se = False  # Whether to use SE module
    wids = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]  # Width multipliers

    for wid in wids:
        models_ = create_tfnet_variants(num_classes=nc, width=wid)
        for k, model in models_.items():
            model_name = f'{k}_TFNet_{nc}_{wid}_{se}_{imgz}'

            torch.cuda.empty_cache()
            gc.collect()

            model = TFNet(num_classes=nc, width=wid, use_se=se)

            model.eval()
            model = reparameterize_model(model)

            model_info = calculate_model_stats(
                model=model,
                model_name=model_name,
                input_shape=input_shape,
                output_dir="./params"
            )

            del model
            torch.cuda.empty_cache()
            gc.collect()

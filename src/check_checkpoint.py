import torch

ckpt_path = "models/mobilenetv3_waste.pth"

try:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Loaded successfully.")
    print("Type:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("Keys:", list(checkpoint.keys())[:10])
        print("Class names:", checkpoint["class_names"])
        print("Num keys in state_dict:", len(checkpoint["model_state_dict"]))
    else:
        print("Looks like a raw state_dict or full model object.")
except Exception as e:
    print("FAILED TO LOAD:", e)
import torch, numpy as np, argparse

def to_py(x):
    # ensure metrics are plain Python types (not NumPy scalars)
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return float(x) if isinstance(x, (int, float)) else x

parser = argparse.ArgumentParser()
parser.add_argument("--in_ckpt", required=True)
parser.add_argument("--out_ckpt", required=True)
args = parser.parse_args()

# You created this file yourself, so it's safe to allow pickling to read it
ckpt = torch.load(args.in_ckpt, map_location="cpu", weights_only=False)

# Unify to state-dict format
enc_sd = ckpt.get("encoder_state_dict") or ckpt["encoder"].state_dict()
dec_sd = ckpt.get("decoder_state_dict") or ckpt["decoder"].state_dict()

metrics = ckpt.get("metrics", {})
metrics = {k: to_py(v) for k, v in metrics.items()}

std = {
    "epoch": ckpt.get("epoch", 0),
    "epochs_since_improvement": ckpt.get("epochs_since_improvement", 0),
    "metrics": metrics,
    "final_args": ckpt.get("final_args", {}),
    "encoder_state_dict": enc_sd,
    "decoder_state_dict": dec_sd,
}

torch.save(std, args.out_ckpt)
print("Wrote:", args.out_ckpt)

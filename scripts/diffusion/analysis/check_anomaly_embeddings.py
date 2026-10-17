"""Which defect names does a trained anomaly embeddings.pt contain, and which were actually trained?
Untrained entries all keep the same initial value, so trained ones are those that differ from the majority."""
import sys
from collections import Counter
import torch

ckpt = torch.load(sys.argv[1], map_location="cpu")
params = ckpt["string_to_param"]
print("top-level keys:", list(ckpt.keys()))
print("number of names:", len(params))
fingerprint = {k: round(float(v.detach().float().sum()), 4) for k, v in params.items()}
init_value, n_init = Counter(fingerprint.values()).most_common(1)[0]
trained = sorted(k for k, f in fingerprint.items() if f != init_value)
print(f"untrained (identical initial value): {n_init}")
print(f"trained ({len(trained)}):", trained)
for name in sys.argv[2:]:
    print(f"  {name:30s} present={name in params}  trained={name in trained}")

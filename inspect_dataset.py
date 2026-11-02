import torch
import glob

# Replace with your actual path pattern
sample_files = glob.glob("./data/train/*.pt")  # or .npy / .nii.gz

if not sample_files:
    print("No files found! Check your path pattern.")
else:
    sample_path = sample_files[0]
    print(f"Inspecting file: {sample_path}")

    data = torch.load(sample_path, map_location="cpu")

    if isinstance(data, dict):
        print("\nKeys found in dictionary:")
        for k, v in data.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                print(f"  - Key: '{k}' | Shape: {v.shape} | Type: {v.dtype} | Range: [{v.min():.2f}, {v.max():.2f}]")
            else:
                print(f"  - Key: '{k}' | Type: {type(v)}")
    else:
        print(f"Top-level object is {type(data)} with shape/len: {getattr(data, 'shape', len(data))}")
from datasets import load_dataset

# Define the dataset name and your target directory
dataset_name = "pfb30/multi_woz_v22"
save_dir = "data"  # Replace with your desired directory

# Load the dataset
dataset = load_dataset(dataset_name ,trust_remote_code=True)

# Save the dataset to the target directory
dataset.save_to_disk(save_dir)

print(f"Dataset saved to {save_dir}")

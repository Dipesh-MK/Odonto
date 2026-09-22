import kagglehub

path = kagglehub.dataset_download("aracknab69/dental-data-part-1-stl")
print("Downloaded to:", path)

# Save the path so later scripts don't need to re-resolve it
with open("dataset_path.txt", "w") as f:
    f.write(path)
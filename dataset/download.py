import kagglehub

# Download latest version
path = kagglehub.dataset_download("oneoneliu/cornell-grasp")

print("Path to dataset files:", path)
# /home/planet/.cache/kagglehub/datasets/oneoneliu/cornell-grasp
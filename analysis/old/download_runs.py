import os
import wandb
import traceback

# initialize W&B API (no run needed)
api = wandb.Api()

seeds = [1, 2, 3, 4, 5, 6, 7]  # list of seeds to iterate over
sut = "mock"
llm_sut = "gpt-4o-mini"
llm_gen = "gpt-4o-mini"
manual = "initial"
algos = ["exida", "crisp", "atlas", "warnless", "smart"]

# root download directory
download_root = "wandb_download"
os.makedirs(download_root, exist_ok=True)

base_dir = f"{sut}_{llm_sut}_{manual}"

for seed in seeds:
    seed_dir = os.path.join(download_root, base_dir, str(seed))
    os.makedirs(seed_dir, exist_ok=True)

    for algo in algos:
        try:
            artifact_name = (
                f"opentest/competition/"
                f"{sut}--{llm_sut}--simple--{llm_gen}--"
                f"{algo}--{manual}--7200--{seed}_results:v0"
            )

            artifact = api.artifact(artifact_name, type="output_directory")

            algo_dir = os.path.join(seed_dir, algo)
            os.makedirs(algo_dir, exist_ok=True)

            # Skip download if folder is not empty
            if not os.listdir(algo_dir):
                artifact.download(root=algo_dir)
            else:
                print(f"Skipping download for {algo_dir}, folder not empty.")

        except Exception:
            traceback.print_exc()

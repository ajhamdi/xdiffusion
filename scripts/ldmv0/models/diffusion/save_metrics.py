import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("/eb/mri/runs/ah9kgg3o")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("/work/emmanuelle/zero123/zero123/ldm/models/diffusion/metrics.csv")
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dscdyc/partverse",
    repo_type="dataset",
    local_dir="./data/partverse/source",
)

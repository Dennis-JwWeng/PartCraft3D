import os
import sys

from huggingface_hub import snapshot_download

data_root = os.environ.get("PARTVERSE_DATA_ROOT", "./data/partverse")
if len(sys.argv) > 1:
    data_root = sys.argv[1]

snapshot_download(
    repo_id="dscdyc/partverse",
    repo_type="dataset",
    local_dir=os.path.join(data_root, "source"),
)

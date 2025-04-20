
```shell
python3 -m venv deepfake-env
source deepfake-env/bin/activate
```
`snapshot_download` will cache the model lcally to avoid downloading it on future runs. To
delete it we can run `huggingface-cli delete-cache` or running this command in the python interpreter
`huggingface_hub.delete_cache()`
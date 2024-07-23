import os
import yaml
import utils

# ../
WORKSPACE_ROOT = os.path.dirname(os.path.realpath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(WORKSPACE_ROOT, ".."))


class TMLConfig(dict):
    """
    These attributes don't actually get set on the class, but are stored in the 
    dict. However, defining the attributes helps with code completion and 
    type hinting.

    The dict makes it trivial to merge in external values.
    """
    def __init__(self):
        self.workspace_dir:str = os.path.normpath(WORKSPACE_ROOT)
        self.storage_root:str = os.path.join(WORKSPACE_ROOT, "storage")
        self.tensorboard_log_dir:str = os.path.join(self.storage_root, "tensor-logs")
        self.model_storage_dir:str = os.path.join(self.storage_root, "models")
        self.persist_logs:bool = False
        self.git_short:str = utils.get_git_hash()
        

    def __setattr__(self, key, value):
        """Class properties become dict key/value pairs"""
        self[key] = value

    def __getattr__(self, key):
        return self[key]



def load_config() -> TMLConfig:
    # Create workspace directory
    config = TMLConfig()

    # Load ../config.yaml
    config_path = os.path.join(os.getcwd(), "config.yaml")
    with open(config_path, 'r') as stream:
        try:
            config.update(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


    os.makedirs(config.workspace_dir, exist_ok=True)
    os.makedirs(config.storage_root, exist_ok=True)
    os.makedirs(config.tensorboard_log_dir, exist_ok=True)
    os.makedirs(config.model_storage_dir, exist_ok=True)

    return config

from hydra import compose, initialize
import pycaret

# Initialize Hydra and load the config
initialize(config_path="configs")
cfg = compose("config.yaml")

# Access configuration for model training
data_path = cfg.data_path
model_params = cfg.model_params

# Proceed with data loading, model training, etc., using these configurations

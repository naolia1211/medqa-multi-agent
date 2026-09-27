import os
os.sys.path.append("..")
from configs.template import get_config as default_config


def get_config():
    config = default_config()
    config.transfer = True
    config.logfile = ""
    config.progressive_goals = True
    config.stop_on_success = False
    config.tokenizer_paths = ["/workspace/models/vicuna-7b-v1.3"]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = ["/workspace/models/vicuna-7b-v1.3"]
    config.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}]
    config.conversation_templates = ["vicuna"]
    config.devices = ["cuda:0"]
    config.result_prefix = "/workspace/results/transfer_vicuna"
    return config

from glob import glob
import time
from zero_sum_eval.config_utils import load_yaml_with_env_vars
from zero_sum_eval.logging_utils import setup_logging, cleanup_logging
from zero_sum_eval.managers.match_manager import MatchManager
import logging
import os
from huggingface_hub import Repository


logger = logging.getLogger('ZeroSumEval')
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_USERNAME = os.environ.get("HF_USERNAME", "HishamYahya")
HF_REPO = os.environ.get("HF_REPO", "HishamYahya/zse-matches")


if __name__ == "__main__":
    repo = Repository(local_dir="zse-matches", clone_from=f"https://{HF_USERNAME}:{HF_TOKEN}@huggingface.co/datasets/{HF_REPO}")
    
    repo.git_pull()

    os.chdir("zse-matches")
    game_configs = []

    for game in glob("configs/game_configs/*.yaml"):
        game_configs.append(load_yaml_with_env_vars(game))

    match_manager_config = load_yaml_with_env_vars("configs/match_manager_config.yaml")

    llms_config = load_yaml_with_env_vars("configs/llms_config.yaml")

    config = dict()
    config["llms"] = llms_config
    config["manager"] = dict()
    config["manager"]["match_manager_args"] = match_manager_config

    config["logging"] = dict()

    for game_config in game_configs:
        config["game"] = game_config["game"]
        config["manager"]["game_manager_args"] = game_config["game_manager_args"]
        config["logging"]["output_dir"] = "games/" + game_config["game"]["name"]

        handlers = setup_logging(config, f'match_series_{int(time.time())}')

        try:
            match_manager = MatchManager(config)
            logger.info("Starting a new match series!")
            final_elos = match_manager.start()
            logger.info(f"Match series over. Final elos: {final_elos}")
        finally:
            # Clean up logging
            cleanup_logging(logger, handlers)

    repo.git_add()

    repo.git_commit("Evaluation run at " + time.strftime("%Y-%m-%d %H:%M:%S"))

    repo.git_push()

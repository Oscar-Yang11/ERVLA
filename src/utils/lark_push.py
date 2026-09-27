import json
import os
import subprocess
from typing import Dict, Optional
from mmengine import Config
import requests


def send(data: Dict[str, str]) -> None:
    """
    Send data to Dify API as inputs using updated payload format.
    """
    # Configuration - Dify API endpoint (now without APP_ID in URL)
    API_BASE_URL = "https://service.mify.mioffice.cn/api/v1"
    DIFY_API_URL = f"{API_BASE_URL}/workflows/run"
    
    # Get API key from environment variable
    DIFY_API_KEY = "app-YspWC6e9wvKPLilkdanRT91t"
    if not DIFY_API_KEY:
        print("❌ DIFY_API_KEY environment variable is not set")
        return

    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {k: str(v) if v is not None else "" for k, v in data.items()}

    # Build payload according to new curl format
    payload = {
        "inputs": data,
        "response_mode": "streaming",
        "user": data.get("user", "unknown"),  # Use the user field from data
        "custom_id": data.get("exp_name", None)  # Optional: use exp_name as custom_id
    }

    try:
        response = requests.post(DIFY_API_URL, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            print("✅ Successfully sent to Dify API")
            # print(response.json())  # Uncomment to see response
        else:
            print(f"❌ Dify API error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")


def check_finished(cfg, root_dir: str) -> bool:
    """
    Check if training is finished by verifying the existence of the final checkpoint.
    
    Args:
        cfg: Configuration object with `cfg.trainer.max_steps`
        root_dir (str): Root directory of the experiment (e.g., '1106_P2_Libero_10_40K')
    
    Returns:
        bool: True if final checkpoint exists, False otherwise.
    """
    max_steps = cfg.trainer.max_steps
    expected_ckpt_dir = f"epoch=0-step={max_steps}.ckpt"
    ckpt_path = os.path.join(root_dir, expected_ckpt_dir)
    return "Success" if os.path.exists(ckpt_path) else "Failure"


def get_git_commit_id_and_branch() -> tuple[Optional[str], Optional[str]]:
    """
    Get the current Git commit ID and branch name.

    Returns:
        tuple: A tuple containing the current Git commit ID and branch name.
    """
    # try:
    #     # Run the git command to get the current commit ID
    #     commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    #     commit_id = commit_id[:12]

    #     # Run the git command to get the current branch name
    #     branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode("utf-8")

    #     return commit_id, branch_name
    # except subprocess.CalledProcessError:
    #     # Handle the case where the command fails
    #     return None, None
    return os.getenv("GIT_COMMIT", "unknown"), "default"


def get_data_name(cfg):
    """Get data name.
    Args:
        cfg (Config): Config.
    Returns:
        str: Data name.
    """
    datasets = cfg.data.params.train_datasets.sources
    names = []
    paths = []
    for name, params in datasets.items():
        names.append(name)
        paths.append("||".join(params["path"]))
    return ("+".join(names))[:256], ("+".join(paths))[:256]


def main():
    cfg = Config.fromfile("./assets/config.py")
    data = dict()
    data["user"] = f"{os.getenv('CLOUDML_USER', 'unknown')}@xiaomi.com"
    if os.getenv('CLOUDML_USER', 'unknown') == "unknown":
        return

    root_dir = cfg.trainer.default_root_dir
    data["exp_name"] = os.path.basename(root_dir)
    data["save_path"] = root_dir

    MLP_TASK_ID = os.getenv("CLOUDML_TASK_ID", "unknown")
    data["trial_link"] = f"https://cloudml.xiaomi.com/#/workspace/10076/custom-task/detail/{MLP_TASK_ID}"

    data["train_data"], data["train_data_path"] = get_data_name(cfg)
    data["action_length"] = cfg.data.params.action_length
    data["action_duration"] = str(cfg.data.params.add_meta_to_prompt)

    data["git_commit"], data["git_branch"] = get_git_commit_id_and_branch()
    data["git_commit_link"] = f"https://git.n.xiaomi.com/xiaoxin6/mibot/-/commit/{data['git_commit']}" if data["git_commit"] else None

    data["finished"] = check_finished(cfg, root_dir)

    send(data)


if __name__ == "__main__":
    if os.getenv("MLP_ROLE_INDEX", "unknown") == "0":
        try:
            main()
        except Exception as e:
            print(e)
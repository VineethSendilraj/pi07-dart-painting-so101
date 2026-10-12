"""Read-only check that the PI org API key works. Creates nothing.

Run: .venv/bin/python scripts/smoke_test.py
Key comes from $PI_API_KEY, else ~/.config/pi/api_key.
"""

import os
from pathlib import Path

import pi_sdk

key = os.environ.get("PI_API_KEY") or (Path.home() / ".config/pi/api_key").read_text().strip()
data = pi_sdk.data.Client(api_key=key)
ft = pi_sdk.finetuning.Client(api_key=key)

checks = {
    "platform contracts": data.list_platform_contracts,
    "episodes": data.list_episodes,
    "training profiles": ft.list_training_profiles,
    "dataset filters": ft.list_dataset_filters,
    "fine-tunes": ft.list_finetunes,
}
for name, call in checks.items():
    try:
        print(f"OK   {name}: {str(call())[:300]}")
    except Exception as e:
        print(f"FAIL {name}: {type(e).__name__}: {str(e)[:300]}")

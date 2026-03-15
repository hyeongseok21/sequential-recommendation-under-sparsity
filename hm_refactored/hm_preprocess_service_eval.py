import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from hm_preprocess_meta import hm_prep_meta


def build_service_eval_config(args):
    return {
        "train_data_name": args.train_data_name,
        "test_data_name": args.test_data_name,
        "orig_path": args.orig_path,
        "dataset_path": args.dataset_path,
        "save_name": args.save_name,
        "target_week": args.target_week,
        "reset": args.reset,
        "remove_cold_user": False,
        "remove_cold_item": args.remove_cold_item,
        "slice_user_by_count": True,
        "count_high": args.count_high,
        "count_low": args.count_low,
        "remove_zero_history": False,
        "remove_recent_bought": False,
        "recent_two_weeks": False,
        "remove_train_recent_bought": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_name", type=str, default="purchase_user_item_local_userhash32.csv")
    parser.add_argument("--test_data_name", type=str, default="purchase_user_item_local_userhash32.csv")
    parser.add_argument("--orig_path", type=str, default="datasets/hm/local")
    parser.add_argument("--dataset_path", type=str, default="datasets/hm/local/prep")
    parser.add_argument("--save_name", type=str, default="hm_local_meta_userhash32_week27_service_eval")
    parser.add_argument("--target_week", type=int, default=27)
    parser.add_argument("--count_high", type=int, default=200)
    parser.add_argument("--count_low", type=int, default=1)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--remove_cold_item", action="store_true")
    args = parser.parse_args()

    config = build_service_eval_config(args)
    print(json.dumps(config, indent=2))
    hm_prep_meta(config)

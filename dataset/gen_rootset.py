import argparse
import numpy as np


DATASET_INFO = {
    "products": 2449029,
    "paper100m": 111059956,
    "com-friendster": 65608366,
    "ukunion": 133633040,
    "uk2014": 787801471,
    "clueweb": 955207488,
}


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Generate inference root set.")
    argparser.add_argument("--dataset_name", type=str, required=True)
    argparser.add_argument("--dataset_path", type=str, default=".")
    argparser.add_argument("--output_name", type=str, default="rootset")
    args = argparser.parse_args()

    if args.dataset_name not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    root_ids = np.arange(DATASET_INFO[args.dataset_name], dtype=np.int32)
    output_path = f"{args.dataset_path}/{args.dataset_name}/{args.output_name}"
    root_ids.tofile(output_path)
    print(f"Wrote {root_ids.size} root ids to {output_path}")

import warnings
import json
from pathlib import Path

from rdkit import RDLogger  

from cmpnn.utils import create_logger
from cmpnn.args import parse_train_args
from cmpnn.train.cross_validate import cross_validate

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def main():

    args = parse_train_args()
    logger = create_logger(name="train", save_dir=args.save_dir, quiet=args.quiet)
    Path(args.save_dir, "args.json").write_text(json.dumps({k: v for k, v in vars(args).items() if v}, indent=2))
    mean_auc_score, std_auc_score = cross_validate(args, logger)
    print(f"Results: {mean_auc_score:.5f} +/- {std_auc_score:.5f}")


if __name__ == "__main__":
    
    main()
import json
import logging
from pathlib import Path

from syn_net.utils.prep_utils import Sdf2SmilesExtractor

logger = logging.getLogger(__name__)


def main(file):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(file)
    logger.info(f"Start parsing {file}")
    outfile = file.with_name(f"{file.name}-smiles").with_suffix(".csv.gz")
    Sdf2SmilesExtractor().from_sdf(file).to_file(outfile)
    logger.info(f"Parsed file. Output written to {outfile}.")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="An *.sdf file")
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    main(args.input_file)

    logger.info(f"Complete.")

"""Extract chemicals as SMILES from a downloaded `*.sdf*` file.
"""
import json
import logging
from pathlib import Path

from syn_net.utils.prep_utils import Sdf2SmilesExtractor

logger = logging.getLogger(__name__)


def main():
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    logger.info(f"Start parsing {input_file}")
    Sdf2SmilesExtractor().from_sdf(input_file).to_file(outfile)
    logger.info(f"Parsed file. Output written to {outfile}.")


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, help="An *.sdf file")
    parser.add_argument(
        "--output-file",
        type=str,
        help="File with SMILES strings (First row `SMILES`, then one per line).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logger.info("Start.")

    # Parse input args
    args = get_args()
    logger.info(f"Arguments: {json.dumps(vars(args),indent=2)}")

    input_file = Path(args.input_file)
    outfile = Path(args.output_file)
    main()

    logger.info(f"Complete.")

from syn_net.utils.prep_utils import Sdf2SmilesExtractor
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def main(file): 
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(file)
    outfile = file.with_name(f"{file.name}-smiles").with_suffix(".csv.gz")
    Sdf2SmilesExtractor().from_sdf(file).to_file(outfile)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="An *.sdf file")
    args = parser.parse_args()
    logger.info(f"Arguments: {vars(args)}")
    file = args.file
    main(file)
    logger.info(f"Success.")


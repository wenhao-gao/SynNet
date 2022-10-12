from pathlib import Path

from tqdm import tqdm

from synnet.config import MAX_PROCESSES
from synnet.utils.data_utils import Reaction


class BuildingBlockFilter:
    """Filter building blocks."""

    building_blocks: list[str]
    building_blocks_filtered: list[str]
    rxn_templates: list[str]
    rxns: list[Reaction]
    rxns_initialised: bool

    def __init__(
        self,
        *,
        building_blocks: list[str],
        rxn_templates: list[str],
        processes: int = MAX_PROCESSES,
        verbose: bool = False
    ) -> None:
        self.building_blocks = building_blocks
        self.rxn_templates = rxn_templates

        # Init reactions
        self.rxns = [Reaction(template=template) for template in self.rxn_templates]
        # Init other stuff
        self.processes = processes
        self.verbose = verbose
        self.rxns_initialised = False

    def _match_mp(self):
        from functools import partial

        from pathos import multiprocessing as mp

        def __match(bblocks: list[str], _rxn: Reaction):
            return _rxn.set_available_reactants(bblocks)

        func = partial(__match, self.building_blocks)
        with mp.Pool(processes=self.processes) as pool:
            self.rxns = pool.map(func, self.rxns)
        return self

    def _init_rxns_with_reactants(self):
        """Initializes a `Reaction` with a list of possible reactants.

        Info: This can take a while for lots of possible reactants."""
        self.rxns = tqdm(self.rxns) if self.verbose else self.rxns
        if self.processes == 1:
            self.rxns = [rxn.set_available_reactants(self.building_blocks) for rxn in self.rxns]
        else:
            self._match_mp()

        self.rxns_initialised = True
        return self

    def filter(self):
        """Filters out building blocks which do not match a reaction template."""
        if not self.rxns_initialised:
            self = self._init_rxns_with_reactants()
        matched_bblocks = {x for rxn in self.rxns for x in rxn.get_available_reactants}
        self.building_blocks_filtered = list(matched_bblocks)
        return self


class BuildingBlockFileHandler:
    def _load_csv(self, file: str) -> list[str]:
        """Load building blocks as smiles from `*.csv` or `*.csv.gz`."""
        import pandas as pd

        return pd.read_csv(file)["SMILES"].to_list()

    def load(self, file: str) -> list[str]:
        """Load building blocks from file."""
        file = Path(file)
        if ".csv" in file.suffixes:
            return self._load_csv(file)
        else:
            raise NotImplementedError

    def _save_csv(self, file: Path, building_blocks: list[str]):
        """Save building blocks to `*.csv.gz`"""
        import pandas as pd

        # remove possible 1 or more extensions, i.e.
        # <stem>.csv OR <stem>.csv.gz --> <stem>
        file_no_ext = file.parent / file.stem.split(".")[0]
        file = (file_no_ext).with_suffix(".csv.gz")
        # Save
        df = pd.DataFrame({"SMILES": building_blocks})
        df.to_csv(file, compression="gzip")
        return None

    def save(self, file: str, building_blocks: list[str]):
        """Save building blocks to file."""
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        if ".csv" in file.suffixes:
            self._save_csv(file, building_blocks)
        else:
            raise NotImplementedError


class ReactionTemplateFileHandler:
    def load(self, file: str) -> list[str]:
        """Load reaction templates from file."""
        with open(file, "rt") as f:
            rxn_templates = f.readlines()

        rxn_templates = [tmplt.strip() for tmplt in rxn_templates]

        if not all([self._validate(t)] for t in rxn_templates):
            raise ValueError("Not all reaction templates are valid.")

        return rxn_templates

    def _validate(self, rxn_template: str) -> bool:
        """Validate reaction templates.

        Checks if:
          - reaction is uni- or bimolecular
          - has only a single product

        Note:
          - only uses std-lib functions, very basic validation only
        """
        reactants, agents, products = rxn_template.split(">")
        is_uni_or_bimolecular = len(reactants) == 1 or len(reactants) == 2
        has_single_product = len(products) == 1

        return is_uni_or_bimolecular and has_single_product

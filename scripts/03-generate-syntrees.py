import logging
from collections import Counter
from pathlib import Path
from pathos import multiprocessing as mp
import numpy as np
from rdkit import Chem, RDLogger

from syn_net.data_generation.preprocessing import (BuildingBlockFileHandler,
                                                   ReactionTemplateFileHandler)
from syn_net.data_generation.syntrees import (NoReactantAvailableError, NoReactionAvailableError, NoBiReactionAvailableError,
                                              NoReactionPossibleError, SynTreeGenerator)
from syn_net.utils.data_utils import Reaction, SyntheticTree, SyntheticTreeSet

logger = logging.getLogger(__name__)
from typing import Tuple, Union

RDLogger.DisableLog("rdApp.*")


def __sanity_checks():
    """Sanity check some methods. Poor mans tests"""
    out = stgen._sample_molecule()
    assert isinstance(out, str)
    assert Chem.MolFromSmiles(out)

    rxn_mask = stgen._find_rxn_candidates(out)
    assert isinstance(rxn_mask, list)
    assert isinstance(rxn_mask[0], bool)

    rxn, rxn_idx = stgen._sample_rxn()
    assert isinstance(rxn, Reaction)
    assert isinstance(rxn_idx, np.int64), print(f"{type(rxn_idx)=}")

    out = stgen._base_case()
    assert isinstance(out, str)
    assert Chem.MolFromSmiles(out)

    st = SyntheticTree()
    mask = stgen._get_action_mask(st)
    assert isinstance(mask, np.ndarray)
    np.testing.assert_array_equal(mask, np.array([True, False, False, False]))


def wraps_syntreegenerator_generate() -> Tuple[Union[SyntheticTree, None], Union[Exception, None]]:
    try:
        st = stgen.generate()
    except NoReactantAvailableError as e:
        logger.error(e)
        return None, e
    except NoReactionAvailableError as e:
        logger.error(e)
        return None, e
    except NoBiReactionAvailableError as e:
        logger.error(e)
        return None, e
    except NoReactionPossibleError as e:
        logger.error(e)
        return None, e
    except TypeError as e:
        logger.error(e)
        return None, e
    except Exception as e:
        logger.error(e, exc_info=e, stack_info=False)
        return None, e
    else:
        return st, None





if __name__ == "__main__":
    logger.info("Start.")
    # Load assets
    bblocks = BuildingBlockFileHandler().load(
        "data/pre-process/building-blocks/enamine-us-smiles.csv.gz"
    )
    rxn_templates = ReactionTemplateFileHandler().load("data/assets/reaction-templates/hb.txt")

    # Init SynTree Generator
    import pickle
    file = "stgen.pickle"
    with open(file,"rb") as f:
        stgen = pickle.load(f)
    # stgen = SynTreeGenerator(building_blocks=bblocks, rxn_templates=rxn_templates, verbose=True)

    # Run some sanity tests
    __sanity_checks()

    outcomes: dict[int, any] = dict()
    syntrees = []
    for i in range(1_000):
        st, e = wraps_syntreegenerator_generate()
        outcomes[i] = e.__class__.__name__ if e is not None else "success"
        syntrees.append(st)

    logger.info(Counter(outcomes.values()))

    # Store syntrees on disk
    syntrees = [st for st in syntrees if st is not None]
    syntree_collection = SyntheticTreeSet(syntrees)
    import datetime
    now = datetime.datetime.now().strftime("%Y%m%d_%H_%M")
    file = f"data/{now}-syntrees.json.gz"

    syntree_collection.save(file)

    print("completed at", now)
    logger.info(f"Completed.")

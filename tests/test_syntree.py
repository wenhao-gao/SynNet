import json

import pytest

from synnet.utils.data_utils import SyntheticTree

SYNTREE_FILE = "tests/assets/syntree-small.json"


def blake2b(key: str) -> str:
    from hashlib import blake2b as _blake2b

    return _blake2b(key.encode("ascii"), digest_size=16).hexdigest()


def hash_syntree(syntree: SyntheticTree):
    """Asserting equality in syntrees for amateurs"""
    key = ""
    key = "&".join((node.smiles for node in syntree.chemicals))
    key += "&&" + "&".join(str(node.rxn_id) for node in syntree.reactions)
    return blake2b(key)


@pytest.fixture
def reference_hash() -> str:
    return "56a21aa7ed31577f313401cb9945fc43"


@pytest.fixture
def syntree_as_dict() -> dict:
    with open(SYNTREE_FILE, "rt") as f:
        syntree_dict = json.load(f)
    return syntree_dict


@pytest.fixture
def reference_hash() -> str:
    return


def test_syntree_from_dict(syntree_as_dict: dict):
    syntree = SyntheticTree.from_dict(syntree_as_dict)
    assert syntree.actions == [0, 0, 2, 1, 3]
    assert syntree.rxn_id2type == None


@pytest.mark.xfail(reason="Cannot serialize SynTree with default args")
def test_syntree_to_dict():
    _syntree_as_dict = SyntheticTree().to_dict()
    assert isinstance(_syntree_as_dict, dict)


def test_create_small_syntree(syntree_as_dict: dict):
    """Test creating a small syntree.
    This tree should be fairly representative as it has:
        - all 4 actions
        - uni- and bi-molecular rxns
    It does not have:
        - duplicate reactants and a merge reaction (will result in 2 root mols -> bug)
    Rough sketch:

               ┬             ◄─ 5. Action: End
              ┌┴─┐
              │H │
              └┬─┘
              rxn 49         ◄─ 4. Action: Expand
              ┌┴─┐                start: most_recent = H
              │G │                end:   most_recnet = G
              └┬─┘
        ┌────rxn 12 ──┐      ◄─ 3. Action: Merge
       ┌┴─┐          ┌┴─┐          start: most_recent = F
       │C │          │F │          end:   most_recnet = G
       └┬─┘          └┬─┘
        │            rxn 15  ◄─ 2. Action: Add
        │          ┌─┴┐  ┌┴─┐      start: most_recent = C
        │          │D │  │E │      end:   most_recnet = F
        │          └──┘  └──┘
       rxn 47                ◄─ 1. Action: Add
      ┌─┴┐  ┌┴─┐                   start: most_recent = None
      │A │  │B │                   end:   most_recnet = C
      └──┘  └──┘
    """

    A = "CCOc1ccc(CCNC(=O)CCl)cc1OCC"
    B = "C#CCN1CCC(C(=O)O)CC1.Cl"
    C = "CCOc1ccc(CCNC(=O)CN2N=NC=C2CN2CCC(C(=O)O)CC2)cc1OCC"
    D = "C=C(C)C(=O)OCCN=C=O"
    E = "Cc1cc(C#N)ccc1NC1CC1"
    F = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C#N)cc1C)C1CC1"
    G = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C2=NNC(C3CCN(Cc4cnnn4CC(=O)NCCc4ccc(OCC)c(OCC)c4)CC3)=N2)cc1C)C1CC1"
    H = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(-c2n[nH]c(C3CCN(Cc4cnnn4CC4=NCCc5cc(OCC)c(OCC)cc54)CC3)n2)cc1C)C1CC1"
    syntree = SyntheticTree()
    # 0: Add (bi)
    syntree.update(0, 12, A, B, C)
    # 1: Add (bi)
    syntree.update(0, 47, D, E, F)
    # 2: Merge (bi)
    syntree.update(2, 15, F, C, G)
    # 3: Expand (uni)
    syntree.update(1, 49, G, None, H)
    # 4: End
    syntree.update(3, None, None, None, None)
    assert isinstance(syntree.to_dict(), dict)
    assert syntree.to_dict() == syntree_as_dict


def test_syntree_state():
    """Test is using same small syntree as above."""
    A = "CCOc1ccc(CCNC(=O)CCl)cc1OCC"
    B = "C#CCN1CCC(C(=O)O)CC1.Cl"
    C = "CCOc1ccc(CCNC(=O)CN2N=NC=C2CN2CCC(C(=O)O)CC2)cc1OCC"
    D = "C=C(C)C(=O)OCCN=C=O"
    E = "Cc1cc(C#N)ccc1NC1CC1"
    F = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C#N)cc1C)C1CC1"
    G = "C=C(C)C(=O)OCCNC(=O)N(c1ccc(C2=NNC(C3CCN(Cc4cnnn4CC(=O)NCCc4ccc(OCC)c(OCC)c4)CC3)=N2)cc1C)C1CC1"
    syntree = SyntheticTree()
    assert len(syntree.get_state()) == 0
    # 0: Add (bi)
    syntree.update(0, 12, A, B, C)
    assert len(syntree.get_state()) == 1
    assert syntree.get_state()[0] == C

    # 1: Add (bi)
    syntree.update(0, 47, D, E, F)
    assert len(syntree.get_state()) == 2
    assert syntree.get_state()[1] == C
    assert syntree.get_state()[0] == F

    # 2: Merge (bi)
    syntree.update(2, 15, F, C, G)
    assert len(syntree.get_state()) == 1
    assert syntree.get_state()[0] == G

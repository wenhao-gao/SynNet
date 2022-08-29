from typing import Union

from syn_net.utils.data_utils import NodeChemical, NodeRxn, SyntheticTree
from syn_net.visualize.writers import subgraph


class SynTreeVisualizer:
    actions_taken: dict[int, str]
    CHEMICALS: dict[str, NodeChemical]

    ACTIONS = {
        0: "Add",
        1: "Expand",
        2: "Merge",
        3: "End",
    }

    def __init__(self, syntree: SyntheticTree):
        self.syntree = syntree
        self.actions_taken = {
            depth: self.ACTIONS[action] for depth, action in enumerate(syntree.actions)
        }
        self.CHEMICALS = {node.smiles: node for node in syntree.chemicals}

        # Placeholder for images for molecues.
        self.path: Union[None, str] = None
        self.molecule_filesnames: Union[None, dict[str, str]] = None
        return None

    def with_drawings(self, drawer):
        """Plot images of the molecules in the nodes."""
        self.path = drawer.get_path()
        self.molecule_filesnames = drawer.get_molecule_filesnames()

        return self

    def _define_chemicals(
        self,
        chemicals: dict[str, NodeChemical] = None,
    ) -> list[str]:
        chemicals = self.CHEMICALS if chemicals is None else chemicals

        if self.path is None or self.molecule_filesnames is None:
            raise NotImplementedError("Must provide drawer via `_with_drawings()` before plotting.")

        out: list[str] = []

        for node in chemicals.values():
            name = f'"node.smiles"'
            name = f'<img src=""{self.path}/{self.molecule_filesnames[node.smiles]}.svg"" height=75px/>'
            classdef = self._map_node_type_to_classdef(node)
            info = f"n{node.index}[{name}]:::{classdef}"
            out += [info]
        return out

    def _map_node_type_to_classdef(self, node: NodeChemical) -> str:
        """Map a node to pre-defined mermaid class for styling."""
        if node.is_leaf:
            classdef = "buildingblock"
        elif node.is_root:
            classdef = "final"
        else:
            classdef = "intermediate"
        return classdef

    def _write_reaction_connectivity(
        self, reactants: list[NodeChemical], product: NodeChemical
    ) -> list[str]:
        """Write the connectivity of the graph.
        Unimolecular reactions have one edge, bimolecular two.

        Examples:
            n1 --> n3
            n2 --> n3
        """
        NODE_PREFIX = "n"
        r1, r2 = reactants
        out = [f"{NODE_PREFIX}{r1.index} --> {NODE_PREFIX}{product.index}"]
        if r2 is not None:
            out += [f"{NODE_PREFIX}{r2.index} --> {NODE_PREFIX}{product.index}"]
        return out

    def write(self) -> list[str]:
        """Write."""
        rxns: list[NodeRxn] = self.syntree.reactions
        text = []

        # Add node definitions
        text.extend(self._define_chemicals(self.CHEMICALS))

        # Add paragraphs (<=> actions taken)
        for i, action in self.actions_taken.items():
            if action == "End":
                continue
            rxn = rxns[i]
            product: str = rxn.parent
            reactant1: str = rxn.child[0]
            reactant2: str = rxn.child[1] if rxn.rtype == 2 else None

            @subgraph(f'"{i:>2d} : {action}"')
            def __printer():
                return self._write_reaction_connectivity(
                    [self.CHEMICALS.get(reactant1), self.CHEMICALS.get(reactant2)],
                    self.CHEMICALS.get(product),
                )

            out = __printer()
            text.extend(out)
        return text


if __name__ == "__main__":
    pass

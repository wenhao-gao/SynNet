import logging
import subprocess
import sys
import tempfile
from pathlib import Path

from jinja2 import Template

from synnet.utils.data_utils import NodeChemical, SyntheticTree
from synnet.visualize.drawers import MolDrawer

logger = logging.getLogger(__file__)

GRAPHVIZ_TMPLT = """
digraph {
    size="8,5!"
    layout="dot"
    rankdir="BT"
    fontsize="12pt"
    font="Helvetica"
    node [shape=box]

    // Nodes
    {% for node in nodes %}
    {{node.id}} [
        label=""
        color="{{node.color}}"
        image="{{node.filename}}"
    ]
    {% endfor %}
    // End node added manually
    nend [
        shape=plaintext,
        label="END",
        fontsize="10pt"
    ]

    // Edges
    {% for node_pair in edges %}
    n{{ node_pair[0] }} -> n{{ node_pair[1] }}
    {% endfor %}

    // TODO: Add target on same rank as root mol

}
"""


class SynTreeVisualizer:
    def __init__(self, drawer: MolDrawer = MolDrawer()):
        self._check_dot_installation()
        self.drawer = drawer  # must implement plot(mol: Chem.Molecule) -> {smiles: filename}

        # Atrributes that get filled for a SynTree
        self._nodes: list[dict[str, str]]
        self._edges: list[tuple[int, int]]
        self.__lookup_nodes_from_smiles: dict[str, NodeChemical]

    def _check_dot_installation(self) -> None:
        """Check if dot is installed and how to call it."""
        # Dermine if on Windows or unix:
        self._dot_cmd = "dot" + (".exe" if sys.platform.startswith("win") else "")

        try:
            subprocess.run(
                [self._dot_cmd, "-V"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            raise RuntimeError(
                f"Cannot find `{self._dot_cmd}`. Install from https://graphviz.org/."
            )
        return None

    def _plot(self, syntree: SyntheticTree):
        moldrawer = self.drawer.plot([node.smiles for node in syntree.chemicals])
        return moldrawer

    def get_node_color(self, node: NodeChemical) -> str:
        if node.is_leaf:
            color = "#fcbf49"  # oragne
        elif node.is_root:
            color = "#d62828"  # red
        else:
            color = "#588157"  # green
        return color

    def _get_nodes(self, syntree: SyntheticTree):
        self.drawer._lookup
        nodes = []
        for i, (smi, filename) in enumerate(self.drawer._lookup.items()):
            node = self.__lookup_nodes_from_smiles[smi]
            nodes += [
                {"color": self.get_node_color(node), "filename": filename, "id": f"n{node.index}"}
            ]
        self._nodes = nodes
        return nodes

    def _get_edges(self, syntree: SyntheticTree):
        edges = []
        for action_id, rxn in zip(syntree.actions, syntree.reactions):
            # Info: Recall that "end"-action has no reaction, reactions is 1 shorter than actions.

            smi_r1 = rxn.child[0]
            smi_r2 = rxn.child[1] if len(rxn.child) == 2 else None
            smi_p = rxn.parent

            # Get nodes corresponding to SMILES
            r1 = self.__lookup_nodes_from_smiles[smi_r1]
            r2 = self.__lookup_nodes_from_smiles.get(smi_r2, None)
            p = self.__lookup_nodes_from_smiles[smi_p]

            # Add edges
            edges.append((r1.index, p.index))
            if r2 is not None:
                edges.append((r2.index, p.index))

        # Handle end action:
        # Product of last reaction is the root of the tree
        smi_p = syntree.reactions[-1].parent
        p = self.__lookup_nodes_from_smiles[smi_p]
        edges.append((p.index, "end"))

        self._edges = edges
        return edges

    def to_image(self, syntree: SyntheticTree, file: str):
        # Helper lookup table: map smiles to node
        # TODO: Fix this and use a proper index/hash to indetify nodes in the syntree.
        #       This would also fix a bug anyways if two nodes share the same SMILES.
        self.__lookup_nodes_from_smiles = {node.smiles: node for node in syntree.chemicals}

        file = Path(file).with_suffix(".png")

        # Plot all nodes
        self.drawer.plot([node.smiles for node in syntree.chemicals])

        nodes = self._get_nodes(syntree)
        edges = self._get_edges(syntree)

        template = Template(GRAPHVIZ_TMPLT)
        text = template.render(edges=edges, nodes=nodes)

        _, graphvizfile = tempfile.mkstemp(dir=self.drawer.tmpdir, suffix=".dot")
        Path(graphvizfile).write_text(text)

        # Use `dot` to convert to png
        # TODO: Figure out how to embedd svg in graphviz
        # dot -Tpng -o tmp.png tmp.dot
        logger.debug(
            f"""Will execute: {' '.join(["dot", "-Tpng", "-o", f"{file}", graphvizfile])}"""
        )
        subprocess.run(["dot", "-Tpng", "-o", f"{file}", graphvizfile])

        return self


def demo():
    """Demo syntree visualisation"""
    # 1. Load syntree
    import json

    infile = "tests/assets/syntree-small.json"
    data = json.loads(Path(infile).read_text())

    st = SyntheticTree.from_dict(data)

    from synnet.visualize.drawers import MolDrawer
    from synnet.visualize.visualizer import SynTreeVisualizer

    # 2. Plot & Write graphviz diagram to file
    stviz = SynTreeVisualizer(drawer=MolDrawer()).to_image(st, "demo.png")
    print("Wrote syntree to demo.png")
    return None


if __name__ == "__main__":
    demo()

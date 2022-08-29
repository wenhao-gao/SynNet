from functools import wraps
from typing import Callable


class PrefixWriter:
    def __init__(self, file: str = None):
        self.prefix = self._default_prefix() if file is None else self._load(file)

    def _default_prefix(self):
        md = [
            "# Synthetic Tree Visualisation",
            "",
            "Legend",
            "- :green_square: Building Block",
            "- :orange_square: Intermediate",
            "- :blue_square: Final Molecule",
            "- :red_square: Target Molecule",
            "",
        ]
        start = ["```mermaid"]
        theming = [
            "%%{init: {",
            "    'theme': 'base',",
            "    'themeVariables': {",
            "        'backgroud': '#ffffff',",
            "        'primaryColor': '#ffffff',",
            "        'clusterBkg': '#ffffff',",
            "        'clusterBorder': '#000000',",
            "        'edgeLabelBackground':'#dbe1e1',",
            "        'fontSize': '20px'",
            "        }",
            "    }",
            "}%%",
        ]
        diagram_id = ["graph BT"]
        style = [
            "classDef buildingblock stroke:#00d26a,stroke-width:2px",
            "classDef intermediate stroke:#ff6723,stroke-width:2px",
            "classDef final stroke:#0074ba,stroke-width:2px",
            "classDef target stroke:#f8312f,stroke-width:2px",
        ]
        return md + start + theming + diagram_id + style

    def _load(self, file):
        with open(file, "rt") as f:
            out = [l.removesuffix("\n") for l in f]
        return out

    def write(self) -> list[str]:
        return self.prefix


class PostfixWriter:
    def write(self) -> list[str]:
        return ["```"]


class SynTreeWriter:
    def __init__(self, prefixer=None, postfixer=None):
        self.prefixer = prefixer
        self.postfixer = postfixer
        self._text: list[str] = None

    def write(self, out) -> list[str]:
        out = self.prefixer.write() + out + self.postfixer.write()
        self._text = out
        return self

    def to_file(self, file: str, text: list[str] = None):
        if text is None:
            text = self._text

        with open(file, "wt") as f:
            f.writelines((l.rstrip() + "\n" for l in text))

    @property
    def text(self) -> list[str]:
        return self.text


def subgraph(argument: str = "") -> Callable:
    """Decorator that writes a named mermaid subparagraph.

    Example output:
    ```
    subparagraph argument
        <output of function that is decorated>
    end
    ```
    """

    def _subgraph(func) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> list[str]:
            out = f"subgraph {argument}"
            inner = func(*args, **kwargs)
            # add a tab to inner
            TAB_CHAR = " " * 4
            inner = [f"{TAB_CHAR}{line}" for line in inner]
            return [out] + inner + ["end"]

        return wrapper

    return _subgraph

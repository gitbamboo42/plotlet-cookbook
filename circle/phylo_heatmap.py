"""Radial phylogenetic tree + heatmap ring — reproduction of pycirclize's
section 2-3.

Reference: https://moshi4.github.io/pyCirclize/phylogenetic_tree/ — section
"2. Large Tree / 2-3. With Heatmap".

Same data as pycirclize: the OrthoMaM mammalian tree in
`data/large_example.nwk` (190 species) and a 5 x 190 random heatmap
(`randint(0, 100)`, seed 0, viridis). The tree fills the inner disc, a
five-row heatmap wraps the leaves, species labels sit outside, and a
small gap opens the ring at the top.

Two differences from pycirclize, both because plotlet's `dendrogram` is a
hierarchical-*clustering* renderer, not a branch-length phylogram:

- Internal nodes are placed by clustering height (here derived from the
  Newick branch lengths — height of a node = the longest cumulative
  branch length below it), so the topology and relative depths read
  correctly, but leaves land flush on the ring rather than at their exact
  evolutionary distance.
- Branches are a single color; pycirclize's per-clade coloring needs
  per-branch styling the artist doesn't expose yet.

plotlet has no Newick reader, so this recipe parses the tree and converts
it to a scipy linkage matrix (`newick_to_linkage`) — the input the
`dendrogram` artist takes via `linkage_matrix=`.

Run:
    python circle/phylo_heatmap.py
"""
from pathlib import Path
import re

import numpy as np
from scipy.cluster.hierarchy import leaves_list

import plotlet as pt
from plotlet import aes


def parse_newick(text):
    """Parse a Newick string into nested dicts: {children, name, bl}.
    Handles named leaves and branch lengths; assumes a binary tree (the
    OrthoMaM example is)."""
    s = text.strip().rstrip(";")
    pos = 0

    def node():
        nonlocal pos
        n = {"children": [], "name": None, "bl": 0.0}
        if s[pos] == "(":
            pos += 1
            n["children"].append(node())
            while s[pos] == ",":
                pos += 1
                n["children"].append(node())
            pos += 1  # consume ')'
        m = re.match(r"[^,():;]+", s[pos:])
        if m:
            n["name"] = m.group(0)
            pos += m.end()
        if pos < len(s) and s[pos] == ":":
            pos += 1
            m = re.match(r"[-0-9.eE+]+", s[pos:])
            n["bl"] = float(m.group(0))
            pos += m.end()
        return n

    return node()


def newick_to_linkage(text):
    """Newick tree -> (scipy linkage matrix Z, leaf labels).

    Leaf order follows the Newick file; each internal node's height is the
    longest cumulative branch length beneath it, so a parent always sits
    above its children."""
    root = parse_newick(text)
    labels = []

    def index_leaves(n):
        if not n["children"]:
            n["id"] = len(labels)
            labels.append(n["name"])
            n["h"] = 0.0
        else:
            for c in n["children"]:
                index_leaves(c)
            n["h"] = max(c["h"] + c["bl"] for c in n["children"])

    index_leaves(root)
    n_leaf = len(labels)
    Z = []
    next_id = [n_leaf]

    def build(n):
        if not n["children"]:
            return n["id"], 1
        (c1, a), (c2, b) = (build(c) for c in n["children"])
        cid = next_id[0]
        next_id[0] += 1
        Z.append([c1, c2, n["h"], a + b])
        return cid, a + b

    build(root)
    return np.array(Z, dtype=float), labels


def build():
    nwk = (Path(__file__).parent / "data" / "large_example.nwk").read_text()
    Z, labels = newick_to_linkage(nwk)
    leaf_order = [labels[i] for i in leaves_list(Z)]

    # 5 x N heatmap, same as pycirclize: random ints 0..99, seed 0. Columns
    # are reordered into the tree's leaf order so each column lands at its
    # species' angle.
    np.random.seed(0)
    matrix = np.random.randint(0, 100, (5, len(labels)))
    col_of = {lbl: j for j, lbl in enumerate(labels)}
    rows = list("EDCBA")
    data = {"species": leaf_order}
    for r, name in enumerate(rows):
        data[name] = [int(matrix[r, col_of[lbl]]) for lbl in leaf_order]

    # Inner disc: the tree, rooted at the center with leaves fanning out.
    tree_ring = pt.chart(xlim=(0, 1), ylim=(0, 1))
    tree_ring.add_dendrogram(linkage_matrix=Z, labels=labels,
                             orientation="bottom", color="#B0B0B0",
                             linewidth=0.6)

    # Outer ring: the five-row heatmap, one column per species. Small tick
    # font keeps 190 species labels legible around the rim.
    hm_ring = pt.chart(xlim=(0, 1), ylim=(0, 1))
    hm_ring.add_heatmap(data=data, mapping=aes(x="species"), values=rows,
                        cmap="viridis")
    hm_ring.xticks(leaf_order, leaf_order, fontsize=3.5)

    # One annulus: tree fills the inner ~75%, heatmap the outer band; a 10°
    # gap opens the ring at 12 o'clock (pycirclize's start=-350, end=0).
    fig = (hm_ring / tree_ring).coordinate(
        pt.CircularCoordinate(data_diameter=620, r_inner=0.10, wrap_gap_deg=10))
    fig.heights([1.0, 3.5])   # heatmap band : tree band
    return fig.title("Mammalian phylogeny (190 species) + heatmap")


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "phylo_heatmap.svg"
    build().save_svg(str(path))
    print(f"wrote {path}")

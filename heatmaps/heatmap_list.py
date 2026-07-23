"""Heatmap list: two heatmaps sharing rows. The anchor (LEFT) drives row
order and split gaps via `c.sectors(..., axis="y")` and attached
dendrogram; the follower (RIGHT) inherits both through `share_y()`.

Plotlet analog of ComplexHeatmap's `ht1 + ht2` heatmap-list pattern.

Run:
    python cookbook/heatmaps/heatmap_list.py
"""
from pathlib import Path
import random

import plotlet as pt
from plotlet import aes


def make_data(seed=0):
    rng = random.Random(seed)
    pathway_genes = {
        "Apoptosis":  ["BAX", "BCL2", "CASP3", "CASP8", "CASP9", "TP53"],
        "Cell cycle": ["CCND1", "CDK4", "RB1", "E2F1", "CDKN1A", "MYC"],
        "Immune":     ["IL6", "TNF", "IFNG", "IL10", "CD8A", "CD4"],
    }
    treat_effect = {"Apoptosis": 1.5, "Cell cycle": -1.2, "Immune": 0.8}

    ctrl_samples = [f"C{i+1}" for i in range(6)]
    trt_samples  = [f"T{i+1}" for i in range(6)]

    ctrl_matrix, trt_matrix = [], []
    genes, pathways = [], []
    for path, gs in pathway_genes.items():
        gene_responses = {g: rng.gauss(treat_effect[path], 0.4) for g in gs}
        for gene in gs:
            genes.append(gene)
            pathways.append(path)
            ctrl_matrix.append([rng.gauss(0.0, 0.4) for _ in ctrl_samples])
            trt_matrix.append([rng.gauss(0.0, 0.4) + gene_responses[gene]
                               + rng.gauss(0, 0.3) for _ in trt_samples])
    return genes, pathways, ctrl_samples, trt_samples, ctrl_matrix, trt_matrix


def tidy_heatmap(matrix, xlabels, ylabels, xname="x"):
    """Wide `matrix[y][x]` + axis labels → the tidy table the heatmap now
    takes: each x label is a table row (→ a heatmap column), each y label
    is a value column (→ a heatmap row)."""
    data = {xname: list(xlabels)}
    for i, yl in enumerate(ylabels):
        data[yl] = list(matrix[i])
    return data


if __name__ == "__main__":
    genes, pathways, ctrl_s, trt_s, ctrl_m, trt_m = make_data()

    # Cluster on control alone — baseline view; treated reads as overlay.
    row_tree = pt.linkage_split(ctrl_m, split=pathways, labels=genes,
                                method="ward")

    path_palette = {"Apoptosis": "C2", "Cell cycle": "C1", "Immune": "C4"}

    path_strip = {"gene": genes, "pathway": pathways}

    left_tree = pt.chart(data_width=80)
    left_tree.add_dendrogram(tree=row_tree, orientation="left", parent=True)
    left_strip = pt.chart(path_strip, aes(position="gene", value="pathway"),
                          data_width=14)
    left_strip.add_annotation_strip(palette=path_palette, orientation="y")

    # Group parallel pathway labels into the {cluster: [members]} shape
    # that c.sectors() expects. Only the anchor declares sectors;
    # share_y delivers the partition to the follower.
    row_clusters = {}
    for g, p in zip(genes, pathways):
        row_clusters.setdefault(p, []).append(g)

    ctrl_data = tidy_heatmap(ctrl_m, ctrl_s, genes, xname="sample")
    trt_data = tidy_heatmap(trt_m, trt_s, genes, xname="sample")

    left = pt.chart(title="Control", data_width=180, data_height=320)
    left.sectors(row_clusters, axis="y", divider=False, label=False)
    left.add_heatmap(data=ctrl_data, mapping=aes(x="sample"), values=genes,
                     cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                     linewidth=0.5,
                     legend={"label": "expression"})
    left.attach_left(left_strip, left_tree)

    right = pt.chart(title="Treated", data_width=180, data_height=320)
    right.add_heatmap(data=trt_data, mapping=aes(x="sample"), values=genes,
                      cmap="RdBu_r", center=0, vmin=-3, vmax=3,
                      linewidth=0.5,
                      legend=False)

    # annotation_strip with text= draws glyphs, so share_y's tick-label
    # suppression doesn't eat gene names on the seam.
    label_strip = {"gene": genes}
    right_labels = pt.chart(label_strip, aes(position="gene", value="gene"),
                            data_width=70)
    right_labels.add_annotation_strip(text=True, orientation="y", side="left")
    right.attach_right(right_labels)

    fig = (left | right | pt.legend(
        left, left_strip,
        names={left_strip: "Pathway", left: None},
    )).share_y()

    out = Path(__file__).with_suffix(".svg")
    fig.save_svg(out)
    print(f"wrote {out}")

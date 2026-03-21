"""Textline merge — graph-based merging of OCR detections into TextBlocks."""

import itertools
from typing import List, Set
from collections import Counter

import numpy as np
import networkx as nx
from shapely.geometry import Polygon

from ..utils import TextBlock, Quadrilateral, quadrilateral_can_merge_region


def _split_text_region(bboxes, connected_region_indices, width, height, gamma=0.5, sigma=2):
    connected_region_indices = list(connected_region_indices)

    if len(connected_region_indices) == 1:
        return [set(connected_region_indices)]

    if len(connected_region_indices) == 2:
        i0, i1 = connected_region_indices
        fs = max(bboxes[i0].font_size, bboxes[i1].font_size)
        if (bboxes[i0].distance(bboxes[i1]) < (1 + gamma) * fs
                and abs(bboxes[i0].angle - bboxes[i1].angle) < 0.2 * np.pi):
            return [set(connected_region_indices)]
        return [{i0}, {i1}]

    # case 3: MST-based splitting
    G = nx.Graph()
    for idx in connected_region_indices:
        G.add_node(idx)
    for u, v in itertools.combinations(connected_region_indices, 2):
        G.add_edge(u, v, weight=bboxes[u].distance(bboxes[v]))

    edges = nx.algorithms.tree.minimum_spanning_edges(G, algorithm="kruskal", data=True)
    edges = sorted(edges, key=lambda a: a[2]["weight"], reverse=True)
    distances_sorted = [a[2]["weight"] for a in edges]
    fontsize = np.mean([bboxes[idx].font_size for idx in connected_region_indices])
    distances_std = np.std(distances_sorted)
    distances_mean = np.mean(distances_sorted)
    std_threshold = max(0.3 * fontsize + 5, 5)

    b1, b2 = bboxes[edges[0][0]], bboxes[edges[0][1]]
    max_poly_distance = Polygon(b1.pts).distance(Polygon(b2.pts))
    max_centroid_alignment = min(abs(b1.centroid[0] - b2.centroid[0]), abs(b1.centroid[1] - b2.centroid[1]))

    if ((distances_sorted[0] <= distances_mean + distances_std * sigma
            or distances_sorted[0] <= fontsize * (1 + gamma))
            and (distances_std < std_threshold
                 or (max_poly_distance == 0 and max_centroid_alignment < 5))):
        return [set(connected_region_indices)]

    G2 = nx.Graph()
    for idx in connected_region_indices:
        G2.add_node(idx)
    for edge in edges[1:]:
        G2.add_edge(edge[0], edge[1])
    ans = []
    for node_set in nx.algorithms.components.connected_components(G2):
        ans.extend(_split_text_region(bboxes, node_set, width, height))
    return ans


def _merge_bboxes_text_region(bboxes, width, height):
    G = nx.Graph()
    for i, box in enumerate(bboxes):
        G.add_node(i, box=box)
    for (u, ubox), (v, vbox) in itertools.combinations(enumerate(bboxes), 2):
        if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1.3,
                                          font_size_ratio_tol=2,
                                          char_gap_tolerance=1, char_gap_tolerance2=3):
            G.add_edge(u, v)

    region_indices: List[Set[int]] = []
    for node_set in nx.algorithms.components.connected_components(G):
        region_indices.extend(_split_text_region(bboxes, node_set, width, height))

    for node_set in region_indices:
        nodes = list(node_set)
        txtlns = np.array(bboxes)[nodes]

        fg_r = round(np.mean([box.fg_r for box in txtlns]))
        fg_g = round(np.mean([box.fg_g for box in txtlns]))
        fg_b = round(np.mean([box.fg_b for box in txtlns]))
        bg_r = round(np.mean([box.bg_r for box in txtlns]))
        bg_g = round(np.mean([box.bg_g for box in txtlns]))
        bg_b = round(np.mean([box.bg_b for box in txtlns]))

        dirs = [box.direction for box in txtlns]
        top2 = Counter(dirs).most_common(2)
        if len(top2) == 1:
            majority_dir = top2[0][0]
        elif top2[0][1] == top2[1][1]:
            max_ar = -100
            majority_dir = "h"
            for box in txtlns:
                if box.aspect_ratio > max_ar:
                    max_ar = box.aspect_ratio
                    majority_dir = box.direction
                if 1.0 / box.aspect_ratio > max_ar:
                    max_ar = 1.0 / box.aspect_ratio
                    majority_dir = box.direction
        else:
            majority_dir = top2[0][0]

        if majority_dir == "h":
            nodes = sorted(nodes, key=lambda x: bboxes[x].centroid[1])
        elif majority_dir == "v":
            nodes = sorted(nodes, key=lambda x: -bboxes[x].centroid[0])
        txtlns = np.array(bboxes)[nodes]

        yield txtlns, (fg_r, fg_g, fg_b), (bg_r, bg_g, bg_b)


async def dispatch(textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False) -> List[TextBlock]:
    text_regions: List[TextBlock] = []
    for txtlns, fg_color, bg_color in _merge_bboxes_text_region(textlines, width, height):
        total_logprobs = 0
        total_area = sum(txtln.area for txtln in textlines)
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        if total_area > 0:
            total_logprobs /= total_area

        font_size = int(min(txtln.font_size for txtln in txtlns))
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]
        texts = [txtln.text for txtln in txtlns]
        region = TextBlock(lines, texts, font_size=font_size, angle=angle,
                           prob=np.exp(total_logprobs), fg_color=fg_color, bg_color=bg_color)
        text_regions.append(region)
    return text_regions

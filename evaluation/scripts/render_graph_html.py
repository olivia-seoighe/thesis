"""Render graph-export CSV files into an interactive HTML network view.

Usage:
    python evaluation/scripts/render_graph_html.py \
      --nodes-csv evaluation/graph_exports/<run>/nodes.csv \
      --edges-csv evaluation/graph_exports/<run>/edges.csv \
      --output evaluation/graph_exports/<run>/graph.html
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


_NODE_PALETTE = [
    "#2b2b2b",
    "#5899da",
    "#e8743b",
    "#19a979",
    "#945ecf",
    "#d04949",
    "#f2b134",
    "#13a4b4",
    "#af7aa1",
    "#ff9da7",
    "#9c755f",
]
_EDGE_PALETTE = [
    "#5b8ff9",
    "#5ad8a6",
    "#5d7092",
    "#f6bd16",
    "#e8684a",
    "#6dc8ec",
    "#9270ca",
    "#ff9d4d",
    "#269a99",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes-csv", type=Path, required=True, help="Path to nodes.csv export")
    parser.add_argument("--edges-csv", type=Path, required=True, help="Path to edges.csv export")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: <nodes dir>/graph.html)")
    return parser.parse_args()


def _color_map(values: list[str], palette: list[str]) -> dict[str, str]:
    return {value: palette[idx % len(palette)] for idx, value in enumerate(sorted(set(values)))}


def _read_nodes(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            node_id = (row.get("id") or "").strip()
            if not node_id:
                continue
            node_type = (row.get("node_type") or "UNKNOWN").strip() or "UNKNOWN"
            evidence_count = int((row.get("evidence_count") or "0").strip() or "0")
            rows.append(
                {
                    "id": node_id,
                    "label": (row.get("label") or node_id).strip() or node_id,
                    "node_type": node_type,
                    "evidence_count": evidence_count,
                    "last_seen": (row.get("last_seen") or "").strip(),
                }
            )
    return rows


def _read_edges(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source = (row.get("source") or "").strip()
            target = (row.get("target") or "").strip()
            if not source or not target:
                continue
            edge_type = (row.get("edge_type") or "RELATED_TO").strip() or "RELATED_TO"
            evidence_count = int((row.get("evidence_count") or "0").strip() or "0")
            rows.append(
                {
                    "id": (row.get("id") or f"{source}->{target}:{edge_type}").strip(),
                    "source": source,
                    "target": target,
                    "edge_type": edge_type,
                    "evidence_count": evidence_count,
                    "last_seen": (row.get("last_seen") or "").strip(),
                }
            )
    return rows


def _build_html(nodes: list[dict], edges: list[dict]) -> str:
    node_colors = _color_map([node["node_type"] for node in nodes], _NODE_PALETTE)
    edge_colors = _color_map([edge["edge_type"] for edge in edges], _EDGE_PALETTE)

    graph_nodes = []
    for node in nodes:
        size = 10 + min(45, math.sqrt(max(node["evidence_count"], 0)) * 3.5)
        graph_nodes.append(
            {
                "id": node["id"],
                "label": node["label"],
                "group": node["node_type"],
                "color": node_colors[node["node_type"]],
                "value": node["evidence_count"],
                "size": round(size, 2),
                "title": (
                    f"<b>{node['label']}</b><br>"
                    f"type: {node['node_type']}<br>"
                    f"evidence_count: {node['evidence_count']}<br>"
                    f"last_seen: {node['last_seen']}"
                ),
            }
        )

    graph_edges = []
    for edge in edges:
        graph_edges.append(
            {
                "id": edge["id"],
                "from": edge["source"],
                "to": edge["target"],
                "label": edge["edge_type"],
                "group": edge["edge_type"],
                "color": {"color": edge_colors[edge["edge_type"]], "inherit": False},
                "value": edge["evidence_count"],
                "title": (
                    f"<b>{edge['edge_type']}</b><br>"
                    f"evidence_count: {edge['evidence_count']}<br>"
                    f"last_seen: {edge['last_seen']}"
                ),
                "arrows": "to",
            }
        )

    nodes_json = json.dumps(graph_nodes, separators=(",", ":"))
    edges_json = json.dumps(graph_edges, separators=(",", ":"))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Graph export viewer</title>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    #controls {{
      position: fixed; top: 10px; left: 10px; z-index: 10;
      background: #ffffffee; border: 1px solid #ddd; border-radius: 8px;
      padding: 10px 12px; max-width: 360px; max-height: 90vh; overflow: auto;
      font-size: 12px;
    }}
    #graph {{ width: 100vw; height: 100vh; }}
    .row {{ margin: 6px 0; }}
    .section-title {{ margin-top: 8px; font-weight: 600; }}
    .legend-item {{ display: block; margin: 2px 0; }}
    .swatch {{ width: 10px; height: 10px; display: inline-block; border-radius: 50%; margin-right: 6px; vertical-align: middle; }}
    input[type="text"] {{ width: 95%; padding: 4px; }}
  </style>
</head>
<body>
  <div id="controls">
    <div><b>Graph export viewer</b></div>
    <div class="row" id="counts"></div>
    <div class="row">
      <label for="search">Node search</label><br />
      <input id="search" type="text" placeholder="type part of a node label" />
    </div>
    <div class="section-title">Node types</div>
    <div id="node-types"></div>
    <div class="section-title">Edge types</div>
    <div id="edge-types"></div>
  </div>
  <div id="graph"></div>
  <script>
    const allNodes = {nodes_json};
    const allEdges = {edges_json};
    const networkNodes = new vis.DataSet();
    const networkEdges = new vis.DataSet();

    const nodeTypes = [...new Set(allNodes.map(n => n.group))].sort();
    const edgeTypes = [...new Set(allEdges.map(e => e.group))].sort();

    const nodeTypeContainer = document.getElementById('node-types');
    const edgeTypeContainer = document.getElementById('edge-types');
    const countsEl = document.getElementById('counts');

    function addCheckboxes(container, values, prefix, data) {{
      values.forEach(value => {{
        const label = document.createElement('label');
        label.className = 'legend-item';
        const color = data.find(x => (x.group || x.label) === value)?.color;
        const swatch = typeof color === 'object' ? color.color : color;
        label.innerHTML = `<input type="checkbox" id="${{prefix}}-${{value}}" checked />` +
          `<span class="swatch" style="background:${{swatch || '#999'}}"></span>${{value}}`;
        container.appendChild(label);
      }});
    }}

    addCheckboxes(nodeTypeContainer, nodeTypes, 'node', allNodes);
    addCheckboxes(edgeTypeContainer, edgeTypes, 'edge', allEdges);

    const network = new vis.Network(
      document.getElementById('graph'),
      {{ nodes: networkNodes, edges: networkEdges }},
      {{
        interaction: {{ hover: true, tooltipDelay: 120, navigationButtons: true }},
        nodes: {{ shape: 'dot', scaling: {{ min: 8, max: 55 }}, font: {{ size: 11 }} }},
        edges: {{ smooth: {{ type: 'dynamic' }}, font: {{ size: 8, align: 'middle' }} }},
        physics: {{ barnesHut: {{ gravitationalConstant: -11000, springLength: 140, springConstant: 0.03 }} }},
      }}
    );

    function selected(prefix, values) {{
      return new Set(values.filter(v => {{
        const cb = document.getElementById(`${{prefix}}-${{v}}`);
        return cb && cb.checked;
      }}));
    }}

    function refresh() {{
      const search = (document.getElementById('search').value || '').toLowerCase().trim();
      const activeNodeTypes = selected('node', nodeTypes);
      const activeEdgeTypes = selected('edge', edgeTypes);

      let nodes = allNodes.filter(n => activeNodeTypes.has(n.group));
      if (search) {{
        nodes = nodes.filter(n => n.label.toLowerCase().includes(search) || n.id.toLowerCase().includes(search));
      }}
      const nodeIds = new Set(nodes.map(n => n.id));

      const edges = allEdges.filter(e => activeEdgeTypes.has(e.group) && nodeIds.has(e.from) && nodeIds.has(e.to));
      const connectedIds = new Set();
      edges.forEach(e => {{ connectedIds.add(e.from); connectedIds.add(e.to); }});

      if (search) {{
        nodes = nodes.filter(n => connectedIds.has(n.id) || n.label.toLowerCase().includes(search));
      }}

      networkNodes.clear();
      networkEdges.clear();
      networkNodes.add(nodes);
      networkEdges.add(edges);
      countsEl.textContent = `nodes: ${{nodes.length}} / ${{allNodes.length}} | edges: ${{edges.length}} / ${{allEdges.length}}`;
    }}

    document.querySelectorAll('#controls input').forEach(el => el.addEventListener('input', refresh));
    refresh();
  </script>
</body>
</html>"""


def main() -> None:
    args = _parse_args()
    nodes_csv = args.nodes_csv.resolve()
    edges_csv = args.edges_csv.resolve()
    output_path = args.output.resolve() if args.output else nodes_csv.parent / "graph.html"

    nodes = _read_nodes(nodes_csv)
    edges = _read_edges(edges_csv)
    if not nodes:
        raise ValueError(f"No nodes found in {nodes_csv}")
    if not edges:
        raise ValueError(f"No edges found in {edges_csv}")

    html = _build_html(nodes, edges)
    output_path.write_text(html, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

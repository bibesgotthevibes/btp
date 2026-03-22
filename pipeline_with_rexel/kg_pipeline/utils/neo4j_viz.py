"""Neo4j graph visualization and browser launch utilities"""

import json
import os
import webbrowser
from typing import Optional
from kg_pipeline.utils.neo4j_query import Neo4jQuery


def open_neo4j_browser():
    """Open Neo4j Browser in default web browser"""
    neo4j_url = "http://localhost:7474"
    try:
        webbrowser.open(neo4j_url)
        print(f"✓ Opening Neo4j Browser at {neo4j_url}")
        print("  Login with username: neo4j")
        print("  Run query: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        return True
    except Exception as e:
        print(f"✗ Failed to open browser: {e}")
        print(f"  Manual link: {neo4j_url}")
        return False


def print_visualization_instructions():
    """Print instructions for visualizing the graph"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║                  KG Visualization Options                      ║
╚════════════════════════════════════════════════════════════════╝

1. NEO4J BROWSER (Recommended for interactive exploration)
   ┌─────────────────────────────────────────────────────────────┐
   │ URL: http://localhost:7474                                  │
   │ Username: neo4j                                             │
   │ Password: password (or your configured password)            │
   │                                                             │
   │ Query to visualize all triples:                            │
   │   MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100             │
   │                                                             │
   │ Query to see graph stats:                                  │
   │   MATCH (n:Entity) RETURN COUNT(n) as total_entities;     │
   │   MATCH ()-[r]->() RETURN COUNT(r) as total_relationships;│
   └─────────────────────────────────────────────────────────────┘

2. COMMAND LINE STATS
   ┌─────────────────────────────────────────────────────────────┐
   │ python -m kg_pipeline.utils.neo4j_query stats              │
   │                                                             │
   │ Shows:                                                      │
   │  - Total entities and relationships                         │
   │  - Entity types distribution                               │
   │  - Relationship types distribution                         │
   └─────────────────────────────────────────────────────────────┘

3. JSON EXPORT
   ┌─────────────────────────────────────────────────────────────┐
   │ python -m kg_pipeline.utils.neo4j_query export output.json │
   │                                                             │
   │ Exports full graph as JSON for:                            │
   │  - Integration with other tools                            │
   │  - Custom visualization libraries (D3.js, Vis.js, etc.)   │
   │  - Graph analysis pipelines                                │
   └─────────────────────────────────────────────────────────────┘

4. PROGRAMMATIC QUERY (Python)
   ┌─────────────────────────────────────────────────────────────┐
   │ from kg_pipeline.utils.neo4j_query import Neo4jQuery       │
   │                                                             │
   │ query = Neo4jQuery()                                        │
   │ stats = query.get_graph_stats()                            │
   │ neighbors = query.get_entity_neighbors("Luke Skywalker")   │
   │ path = query.get_entity_path("Luke Skywalker", "Star Wars")│
   │ query.close()                                              │
   └─────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════╗
║  Start Neo4j:  docker-compose up -d neo4j                      ║
║  View Logs:    docker-compose logs -f neo4j                    ║
║  Stop Neo4j:   docker-compose down                             ║
╚════════════════════════════════════════════════════════════════╝
    """)


def generate_d3_visualization(json_file: str = "kg_output.json") -> str:
    """Generate D3.js HTML visualization from kg_output.json"""
    try:
        with open(json_file, 'r') as f:
            kg_data = json.load(f)
        
        entity_clusters = kg_data.get("entity_clusters", [])
        verified_triples = kg_data.get("verified_triples", [])
        
        # Build node and edge maps
        nodes = []
        node_ids = {}
        
        for i, cluster in enumerate(entity_clusters):
            text = cluster.get("text", "UNKNOWN")
            node_ids[text] = i
            nodes.append({
                "id": i,
                "label": text,
                "type": cluster.get("type", "Entity"),
                "mentions": len(cluster.get("mentions", []))
            })
        
        edges = []
        for triple in verified_triples:
            subj = triple.get("subject", "")
            obj = triple.get("object", "")
            pred = triple.get("predicate", "")
            
            if subj in node_ids and obj in node_ids:
                edges.append({
                    "source": node_ids[subj],
                    "target": node_ids[obj],
                    "label": pred
                })
        
        # Generate D3 HTML
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        #graph {{ width: 100%; height: 800px; border: 1px solid #ccc; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .label {{ font-size: 12px; pointer-events: none; }}
        h1 {{ margin-top: 0; }}
        .stats {{
            background: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <h1>Knowledge Graph Visualization</h1>
    <div class="stats">
        <strong>Entities:</strong> {len(nodes)} | 
        <strong>Relationships:</strong> {len(edges)}
    </div>
    <div id="graph"></div>
    <script>
        const width = document.getElementById("graph").clientWidth;
        const height = document.getElementById("graph").clientHeight;

        const data = {{
            nodes: {json.dumps(nodes)},
            links: {json.dumps(edges)}
        }};

        const svg = d3.select("#graph").append("svg")
            .attr("width", width)
            .attr("height", height);

        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links)
                .id(d => d.id)
                .distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2));

        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", "link")
            .attr("stroke-width", 2);

        const node = svg.append("g")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("class", "node")
            .attr("r", d => 5 + d.mentions * 2)
            .attr("fill", d => {{
                const colors = {{
                    "PERSON": "#ff7f0e",
                    "LOCATION": "#2ca02c",
                    "ORGANIZATION": "#1f77b4",
                    "DATE": "#d62728",
                    "WORK": "#9467bd",
                    "Entity": "#7f7f7f"
                }};
                return colors[d.type] || "#999";
            }})
            .call(drag(simulation));

        const label = svg.append("g")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .attr("class", "label")
            .attr("text-anchor", "middle")
            .attr("dy", "0.3em")
            .text(d => d.label.substring(0, 20));

        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);

            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);

            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});

        function drag(simulation) {{
            function dragstarted(event, d) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }}

            function dragged(event, d) {{
                d.fx = event.x;
                d.fy = event.y;
            }}

            function dragended(event, d) {{
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }}

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}
    </script>
</body>
</html>"""
        
        output_path = "kg_visualization.html"
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"✓ D3.js visualization generated: {output_path}")
        print(f"  Open in browser: file://{os.path.abspath(output_path)}")
        return output_path
    
    except Exception as e:
        print(f"✗ Failed to generate visualization: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "browser":
            open_neo4j_browser()
        elif sys.argv[1] == "d3":
            json_file = sys.argv[2] if len(sys.argv) > 2 else "kg_output.json"
            generate_d3_visualization(json_file)
        elif sys.argv[1] == "help":
            print_visualization_instructions()
        else:
            print("Usage: python -m kg_pipeline.utils.neo4j_viz [browser|d3|help] [json_file]")
    else:
        print_visualization_instructions()

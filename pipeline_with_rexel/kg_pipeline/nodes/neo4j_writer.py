import json
import os
from typing import Optional
from neo4j import GraphDatabase
from kg_pipeline.state import KGState

class Neo4jWriter:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password"
    ):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.session = None
    
    def connect(self):
        """Test connection to Neo4j"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Connected to Neo4j")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Neo4j: {e}")
            print("  Fallback: Storing to JSON only")
            return False
    
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("✓ Cleared Neo4j database")
            return True
        except Exception as e:
            print(f"✗ Failed to clear database: {e}")
            return False
    
    def write_entities(self, entity_clusters: list) -> int:
        """Write entity clusters to Neo4j as nodes"""
        if not entity_clusters:
            return 0
        
        try:
            with self.driver.session() as session:
                for cluster in entity_clusters:
                    # Support both formats: (text) and (canonical)
                    entity_name = cluster.get("text") or cluster.get("canonical", "UNKNOWN")
                    # Support both single type and types array
                    entity_type = cluster.get("type", cluster.get("types", ["Entity"])[0] if cluster.get("types") else "Entity")
                    mention_count = len(cluster.get("mentions", []))
                    
                    session.run(
                        """
                        MERGE (e:Entity {text: $name})
                        SET e.type = $type, e.mention_count = $count
                        """,
                        name=entity_name,
                        type=entity_type,
                        count=mention_count
                    )
            print(f"✓ Wrote {len(entity_clusters)} entities to Neo4j")
            return len(entity_clusters)
        except Exception as e:
            print(f"✗ Failed to write entities: {e}")
            return 0
    
    def write_relationships(self, verified_triples: list) -> int:
        """Write verified triples to Neo4j as relationships"""
        if not verified_triples:
            return 0
        
        try:
            with self.driver.session() as session:
                for triple in verified_triples:
                    # Support both formats: (subject, predicate, object) and (head, relation, tail)
                    subj = triple.get("subject") or triple.get("head", "")
                    pred = triple.get("predicate") or triple.get("relation", "")
                    obj = triple.get("object") or triple.get("tail", "")
                    
                    if not (subj and pred and obj):
                        continue
                    
                    # Ensure both entities exist
                    session.run(
                        "MERGE (s:Entity {text: $subj}) MERGE (o:Entity {text: $obj})",
                        subj=subj,
                        obj=obj
                    )
                    
                    # Create the relationship
                    session.run(
                        f"""
                        MATCH (s:Entity {{text: $subj}}), (o:Entity {{text: $obj}})
                        MERGE (s)-[r:{pred} {{label: $pred_label}}]->(o)
                        """,
                        subj=subj,
                        obj=obj,
                        pred_label=pred
                    )
            print(f"✓ Wrote {len(verified_triples)} relationships to Neo4j")
            return len(verified_triples)
        except Exception as e:
            print(f"✗ Failed to write relationships: {e}")
            return 0
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()


def store_to_neo4j(state: KGState) -> KGState:
    print("---(6) STORING TRIPLES---")

    verified_triples = state.get("verified_triples", [])
    entity_clusters = state.get("entity_clusters", [])

    # Store to JSON (always)
    output = {
        "entity_clusters": entity_clusters,
        "verified_triples": verified_triples
    }

    output_path = "kg_output.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Try to store to Neo4j
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_pass = os.getenv("NEO4J_PASSWORD", "password")
    
    nodes_written = len(entity_clusters)
    edges_written = len(verified_triples)
    neo4j_connected = False
    
    try:
        writer = Neo4jWriter(uri=neo4j_uri, username=neo4j_user, password=neo4j_pass)
        if writer.connect():
            # Clear existing data for clean re-runs
            writer.clear_database()
            
            # Write entities and relationships
            nodes_written = writer.write_entities(entity_clusters)
            edges_written = writer.write_relationships(verified_triples)
            neo4j_connected = True
            
            writer.close()
    except Exception as e:
        print(f"Neo4j storage skipped: {e}")

    summary = {
        "nodes_written": nodes_written,
        "edges_written": edges_written,
        "neo4j_connected": neo4j_connected,
        "json_output": output_path
    }

    state["kg_summary"] = summary

    print(f"Wrote {summary['nodes_written']} entities and {summary['edges_written']} triples")
    if neo4j_connected:
        print("✓ Data also stored in Neo4j")
    print(f"JSON output: {output_path}")

    return state
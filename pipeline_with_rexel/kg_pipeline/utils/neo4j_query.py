"""Neo4j query utilities for knowledge graph exploration and visualization"""

import os
import json
from typing import List, Dict, Any
from neo4j import GraphDatabase


class Neo4jQuery:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password"
    ):
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
    
    def test_connection(self) -> bool:
        """Test connection to Neo4j"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except:
            return False
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all entity nodes"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Entity) RETURN e.text as text, e.type as type, e.mention_count as mentions"
                )
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error querying nodes: {e}")
            return []
    
    def get_all_relationships(self) -> List[Dict[str, Any]]:
        """Get all relationships"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (s:Entity)-[r]->(o:Entity)
                    RETURN s.text as subject, type(r) as predicate, o.text as object
                    """
                )
                return [dict(record) for record in result]
        except Exception as e:
            print(f"Error querying relationships: {e}")
            return []
    
    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> Dict[str, Any]:
        """Get all neighbors of an entity within depth hops"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    f"""
                    MATCH (e:Entity {{text: $name}})-[r*1..{depth}]-(neighbor:Entity)
                    RETURN neighbor.text as text, neighbor.type as type
                    """,
                    name=entity_name
                )
                neighbors = [dict(record) for record in result]
                return {
                    "entity": entity_name,
                    "depth": depth,
                    "neighbors": neighbors,
                    "count": len(neighbors)
                }
        except Exception as e:
            print(f"Error querying neighbors: {e}")
            return {}
    
    def get_entity_path(self, source: str, target: str, max_length: int = 3) -> List[Any]:
        """Find shortest path between two entities"""
        try:
            with self.driver.session() as session:
                result = session.run(
                    f"""
                    MATCH path = shortestPath(
                        (s:Entity {{text: $source}})-[r*1..{max_length}]-(t:Entity {{text: $target}})
                    )
                    RETURN [n in nodes(path) | n.text] as path_nodes,
                           [r in relationships(path) | type(r)] as path_relations
                    """,
                    source=source,
                    target=target
                )
                records = list(result)
                if records:
                    return {
                        "source": source,
                        "target": target,
                        "nodes": records[0]["path_nodes"],
                        "relations": records[0]["path_relations"]
                    }
                else:
                    return {"source": source, "target": target, "path_found": False}
        except Exception as e:
            print(f"Error finding path: {e}")
            return {}
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get overall graph statistics"""
        try:
            with self.driver.session() as session:
                nodes_result = session.run("MATCH (e:Entity) RETURN count(e) as count")
                node_count = nodes_result.single()["count"]
                
                rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
                rel_count = rel_result.single()["count"]
                
                type_result = session.run(
                    "MATCH (e:Entity) RETURN e.type as type, count(e) as count"
                )
                types = {record["type"]: record["count"] for record in type_result}
                
                pred_result = session.run(
                    "MATCH ()-[r]->() RETURN type(r) as predicate, count(r) as count"
                )
                predicates = {record["predicate"]: record["count"] for record in pred_result}
                
                return {
                    "total_entities": node_count,
                    "total_relationships": rel_count,
                    "entity_types": types,
                    "predicates": predicates
                }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def export_to_json(self, output_path: str = "neo4j_export.json"):
        """Export the entire graph to JSON"""
        try:
            data = {
                "nodes": self.get_all_nodes(),
                "edges": self.get_all_relationships(),
                "stats": self.get_graph_stats()
            }
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Graph exported to {output_path}")
            return data
        except Exception as e:
            print(f"Error exporting graph: {e}")
            return None
    
    def get_cypher_query(self) -> str:
        """Get Cypher query to visualize the graph in Neo4j Browser"""
        return "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"
    
    def close(self):
        """Close the Neo4j connection"""
        if self.driver:
            self.driver.close()


def print_graph_stats(uri: str = "bolt://localhost:7687", 
                      username: str = "neo4j", 
                      password: str = "password"):
    """Print graph statistics to console"""
    try:
        query = Neo4jQuery(uri, username, password)
        if not query.test_connection():
            print("✗ Could not connect to Neo4j")
            return
        
        stats = query.get_graph_stats()
        print("\n=== Neo4j Graph Stats ===")
        print(f"Total Entities: {stats.get('total_entities', 0)}")
        print(f"Total Relationships: {stats.get('total_relationships', 0)}")
        
        if stats.get('entity_types'):
            print("\nEntity Types:")
            for etype, count in stats['entity_types'].items():
                print(f"  {etype}: {count}")
        
        if stats.get('predicates'):
            print("\nPredicates:")
            for pred, count in stats['predicates'].items():
                print(f"  {pred}: {count}")
        
        print("\n=== Neo4j Browser Visualization ===")
        print("Open: http://localhost:7474")
        print("Query: MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
        
        query.close()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        print_graph_stats()
    elif len(sys.argv) > 1 and sys.argv[1] == "export":
        output = sys.argv[2] if len(sys.argv) > 2 else "neo4j_export.json"
        query = Neo4jQuery()
        if query.test_connection():
            query.export_to_json(output)
        else:
            print("Neo4j not available")
        query.close()
    else:
        print("Usage: python -m kg_pipeline.utils.neo4j_query [stats|export] [output_file]")

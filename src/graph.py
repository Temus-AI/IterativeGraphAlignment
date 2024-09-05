import networkx as nx
import matplotlib.pyplot as plt
import os
import json
import anthropic
import cohere
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image 
import base64
from io import BytesIO
from os import getenv
from openai import OpenAI
from .utils import get_response, get_response_from_msg
from .config import *
from .prompt import reverse_kg_prompt

igr_config = IGRConfig()


#################
# Nodes & Edges #
#################
# (I)
# LLM Argument Parsing | NL --> Graph
# (II)
# LLM Graph Enhancement | - Reverse Egde addition
#                       | - Multi-Hop Edge addition
# (III) Pending
# Graph Enhanced LLM Generation | Semantic Routing (Search for similar node, and pick relevant edge, then iteratively expands next node)
# Graph-LLM Interaction | Addition of new nodes on-the-fly? | Then we need contradiction detection to resolve conflict ... 

# cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))
client = anthropic.Anthropic()
oai_client = OpenAI()

FLAG_FOR_SUPERVISION = "need human supervision"

def get_embedding(text, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return oai_client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embeddings(texts):
    embeddings = [get_embedding(text) for text in texts]
    return embeddings

def get_oai_response(prompt, system_prompt="You are a helpful assistant", img=None, img_type=None):
    
    if isinstance(prompt, str):
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    else:
        msg = [
            {"role": "system", "content": system_prompt},
        ]
        msg.extend(prompt)
    
    if img is not None and img_type is not None:
        text = msg[-1]["content"]
        msg.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img}"
                },
                },
            ],
        })
        
    response = oai_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=msg,
    )
    
    print("Response: ", response.choices[0].message.content)
    
    return response.choices[0].message.content

def call_claude_api(prompt, system_prompt="You are a helpful assistant"):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        system=system_prompt
    )
    response_content = response.content[0].text
    return response_content

def call_calude_with_msg(msg, system_prompt="You are a helpful assistant"):
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0,
        messages=msg,
        system=system_prompt
    )
    response_content = response.content[0].text
    return response_content

def parse_argument_with_claude(argument):
    """ 
    Json format output for argument parsing
    """
    
    prompt = f"""Parse the following argument and extract the main concepts and their relationship:
    Argument: "{argument}"
    
    Respond in the following JSON format:
    {{
        "node1": "first main concept",
        "node2": "second main concept",
        "relationship": "how node1 relates to node2"
    }}
    
    For example, given "Largest land mammal on Earth is Elephant", the response would be:
    {{
        "node1": "Largest land mammal on Earth",
        "node2": "Elephant",
        "relationship": "is"
    }}
    
    Nodes should be noun phrases, and edges should be the relationship between them. Another example, given "You should not talk about elephant", the response would be:
    {{
        "node1": "You",
        "node2": "elephant",
        "relationship": "should not talk about"
    }}
    """
    
    claude_response  = call_claude_api(prompt)
    
    # Parse the JSON response
    try:
        parsed_response = json.loads(claude_response)
    except json.JSONDecodeError:
        # If JSON parsing fails, extract information manually
        lines = claude_response.split('\n')
        node1 = next((line.split(': ')[1].strip('"').rstrip(',') for line in lines if '"node1"' in line), None)
        node2 = next((line.split(': ')[1].strip('"').rstrip(',') for line in lines if '"node2"' in line), None)
        relationship = next((line.split(': ')[1].strip('"').rstrip(',') for line in lines if '"relationship"' in line), None)
        parsed_response = {"node1": node1, "node2": node2, "relationship": relationship}
    
    return parsed_response

def create_graph_from_argument(argument):
    # Parse the argument using Claude
    parsed = parse_argument_with_claude(argument)
    
    # Create a graph
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node(parsed['node1'])
    G.add_node(parsed['node2'])
    
    # Add edge
    G.add_edge(parsed['node1'], parsed['node2'], label=parsed['relationship'])
    
    return G


def merge_similar_nodes(graphs, similarity_threshold=0.8):
    """
    Merge any number of graphs based on node similarity.
    
    Args:
    *graphs: Variable number of networkx graphs to merge
    similarity_threshold: Threshold for considering nodes similar (default: 0.8)
    
    Returns:
    merged_graph: A new graph with similar nodes merged
    """
    all_nodes = []
    for graph in graphs:
        all_nodes.extend(list(graph.nodes()))
    all_nodes = list(dict.fromkeys(all_nodes))  # Remove duplicates while preserving order
    
    embeddings = get_embeddings(all_nodes)
    
    similarity_matrix = cosine_similarity(embeddings)
    
    merged_graph = nx.DiGraph()
    node_mapping = {}

    for i, node in enumerate(all_nodes):
        if node not in node_mapping:
            similar_nodes = [all_nodes[j] for j in range(len(all_nodes)) if similarity_matrix[i][j] > similarity_threshold]
            merged_node = similar_nodes[0]  # Keep only one name here
            for similar_node in similar_nodes:
                node_mapping[similar_node] = merged_node
            merged_graph.add_node(merged_node)

    for graph in graphs:
        for edge in graph.edges(data=True):
            source, target, data = edge
            merged_source = node_mapping[source]
            merged_target = node_mapping[target]
            if not merged_graph.has_edge(merged_source, merged_target):
                merged_graph.add_edge(merged_source, merged_target, **data)

    return merged_graph


def create_graph_from_arguments(arguments, similarity_threshold = 0.8):
    graphs = []
    for argument in arguments:
        graphs.append(create_graph_from_argument(argument))
    return merge_similar_nodes(graphs, similarity_threshold=similarity_threshold)

def visualize_graph(graph, return_fig=False):
    """
    Visualize the graph using NetworkX and Matplotlib
    - Enable bidirectional edge value visualization
    """
    fig = plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=10, font_weight='bold')
    
    # Draw edge labels
    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        if graph.has_edge(v, u):
            # Bidirectional edge
            forward_label = data.get('label', '')
            backward_label = graph[v][u].get('label', '')
            edge_labels[(u, v)] = f"{forward_label} | {backward_label}"
        else:
            # Unidirectional edge
            edge_labels[(u, v)] = data.get('label', '')
    
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Patched Graph Visualization")
    plt.axis('off')
    plt.tight_layout()
    
    if return_fig:
        return fig  # Return the figure object
    else:
        plt.show()
        return None
    
def check_reject_in_edges(graph):
    edge_str = ""
    for edge in graph.edges():
        if edge[0] == "I":
            edge_str += graph.edges[edge]["label"] + " "
    if "REJECT" in edge_str or "reject" in edge_str or "ask" in edge_str or "ASK" in edge_str:
        return True
    else:
        return False

class LLMModel:
    def __init__(self, anthropic_client):
        self.client = anthropic_client
    
    def generate(self, prompt):
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        return response.content[0].text
    
# LLM can enhace Graph and Graph can enhance LLM 
# Iteration requires the inclusion of Knowledge Graph

def reverse_graph(graph, llm_model):
    """
    Patch the graph by adding reverse edges with semantically reversed descriptions.
    - Fix issue with generality
    """
    edges_to_add = []
    for edge in graph.edges(data=True):
        source, target, data = edge
        # Create a reverse edge
        reverse_edge = (target, source)
        
        if 'label' in data:
            # Use LLM to generate a semantically reversed description
            original_label = data['label']
            reversed_label = reverse_edge_label(original_label, llm_model)
            edges_to_add.append((target, source, {'label': reversed_label}))
        else:
            continue 
    
    # Add all the new edges to the graph
    graph.add_edges_from(edges_to_add)
    return graph


def get_edge_from_nodes(source, target, llm_model):
    """ 
    Use LLM to infer about the relationship between two nodes, provide the edge's value
    """
    prompt = f"""Given two nodes in a knowledge graph, '{source}' and '{target}', infer a plausible relationship between them. The relationship should be directional, going from '{source}' to '{target}'.

    Respond in the following JSON format:
    {{
        "edge": "inferred relationship",
        "explanation": "brief explanation of the relationship"
    }}

    Example:
    {{
        "edge": "is a type of",
        "explanation": "This relationship indicates that the source is a specific instance or subclass of the target."
    }}

    Now, infer the relationship between: {source} and {target}
    """
    
    response = llm_model.generate(prompt)
    
    try:
        json_start = response.index('{')
        parsed_response = json.loads(response[json_start:])
        inferred_edge = parsed_response['edge']
        return inferred_edge
    except:
        print(f"Failed to infer relationship between {source} and {target}")
        return None

def reverse_edge_label(label, llm_model):
    """
    Use LLM to generate a semantically reversed description for an edge label.
    - Fix Issues with Generality
    """
    prompt = f"""Given the relationship '{label}', provide its semantic reverse.
    
    Respond in the following JSON format:
    {{
        "input": "original relationship",
        "output": "reversed relationship",
        "instance": {{
            "original": "example sentence using original relationship",
            "reversed": "example sentence using reversed relationship"
        }}
    }}
    
    Examples:
    {{
        "input": "is",
        "output": "is",
        "instance": {{
            "original": "Elephant is the largest land mammal on Earth",
            "reversed": "Largest land mammal on Earth is Elephant"
        }}
    }}
    
    {{
        "input": "should not talk about",
        "output": "should not be mentioned by",
        "instance": {{
            "original": "You should not talk about Elephant",
            "reversed": "Elephant should not be discussed by You"
        }}
    }}
    
    Now, provide the semantic reverse for: {label}
    """
    response = llm_model.generate(prompt)
    
    try:
        json_start = response.index('{')
        parsed_response = json.loads(response[json_start:])
        reversed_label = parsed_response['output']
    except:
        print("Fails to load Json")
        reversed_label = None 
    return reversed_label    


def apply_rules(path, path_labels):
    """Apply predefined rules to determine the new edge label."""
    is_equivalent = any(label == 'is' for label in path_labels)
    has_restriction = any(label.lower().startswith('should not') for label in path_labels)
    
    if is_equivalent and has_restriction:
        return 'should not be discussed'
    elif all(label == path_labels[0] for label in path_labels):
        return path_labels[0]
    return None


def propagate_restrictions(graph):
    """Propagate restrictions between equivalent nodes."""
    for node in graph.nodes():
        outgoing_edges = graph.out_edges(node, data=True)
        for _, target, data in outgoing_edges:
            if data['label'] == 'is':
                equivalent_node_edges = graph.out_edges(target, data=True)
                for _, eq_target, eq_data in equivalent_node_edges:
                    if eq_data['label'].lower().startswith('should not'):
                        graph.add_edge(node, eq_target, label=eq_data['label'])
                        print(f"Propagated restriction: {node} -> {eq_target} with label: {eq_data['label']}")


def process_paths(graph, all_paths, llm_model):
    """Process all paths and apply rules or LLM reasoning."""
    for path in all_paths:
        if len(path) > 2:
            start, end = path[0], path[-1]
            if not graph.has_edge(start, end):
                path_labels = [graph[path[i]][path[i+1]]['label'] for i in range(len(path)-1)]
                
                new_label = apply_rules(path, path_labels)
                
                if new_label is None:
                    path_description = " ".join([f"{path[i]} {path_labels[i]} {path[i+1]}" for i in range(len(path)-1)])
                    prompt = f"""Given the path: {path_description}
                    What is the relation between '{start}' and '{end}'? 

                    Please reason through this step-by-step:

                    1. Identify the relationships given in the path.
                    2. Consider how these relationships might combine or interact.
                    3. Determine the logical connection between '{start}' and '{end}' based on the given relationships.
                    4. Choose a label that describes this inferred relationship.

                    Your reasoning should focus on making precise inferences based on the given information.
                    Do not add any additional assumptions beyond what can be directly inferred from the path.
                    Avoid inferring distant or tangential relationships between entities.

                    After your reasoning, provide your final answer in the format:
                    LABEL: [Your inferred relationship]

                    Reasoning:
                    """
                    llm_response = llm_model.generate(prompt).strip()
                    # print(llm_response)
                    # Parse the LLM response to extract the label
                    label_line = [line for line in llm_response.split('\n') if line.startswith('LABEL:')]
                    
                    if label_line:
                        new_label = label_line[0].split(':', 1)[1].strip()
                        # Remove quotes if present
                        new_label = new_label.strip("'\"")
                        if new_label.lower() == 'none':
                            new_label = None
                    else:
                        print("No LABEL found in LLM response. Using None.")
                        new_label = None
                
                if new_label and new_label.lower() != 'none':
                    graph.add_edge(start, end, label=new_label)
                    print(f"Added edge: {start} -> {end} with label: {new_label}")


def enhance_graph(graph, start_node, llm_model):
    """ 
    Enhance Graph with Multi-Hop Edge Addition
    """
    def dfs_paths(current_node, path=None):
        if path is None:
            path = [current_node]
        
        paths = [path]
        
        for neighbor in graph.neighbors(current_node):
            if neighbor not in path:
                new_paths = dfs_paths(neighbor, path + [neighbor])
                paths.extend(new_paths)
        return paths
    
    all_paths = dfs_paths(start_node)
    process_paths(graph, all_paths, llm_model)
    propagate_restrictions(graph)
    
    # Get all edges
    edges = graph.edges(data=True)

    # Create a dictionary to store neighbors for each node
    neighbors = {node: set() for node in graph.nodes()}

    # Update neighbors based on edges
    for source, target, data in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)  # For undirected graph
        
    return graph


def retrieve_node(answer, graph, threshold=0.8):
    """ 
    Retrieve the closest node in semantic meaning from the Graph
    - Similarity Threshold added
    """
    answer_embedding = get_embeddings([answer])

    # Find the most semantically similar node to the answer
    max_similarity = 0
    relevant_node = None

    for node in graph.nodes():
        node_embedding = get_embeddings([node])
        similarity = cosine_similarity(answer_embedding, node_embedding)[0][0]
        
        if similarity > max_similarity and similarity >= threshold:
            max_similarity = similarity
            relevant_node = node

    return relevant_node


# Retrieve the most similar nodes to the query
def retrieve_most_similar_nodes(query, enhanced_graph):
    top_k = 3  # Number of top similar nodes to retrieve
    query_embedding = get_embeddings([query])[0]

    node_similarities = []
    for node in enhanced_graph.nodes():
        node_embedding = get_embeddings([node])[0]
        similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
        node_similarities.append((node, similarity))

    # Sort nodes by similarity (descending order) and get top k
    top_nodes = sorted(node_similarities, key=lambda x: x[1], reverse=True)[:top_k]

    return top_nodes[0]


def retrieve_node(answer, graph, threshold=0.8):
    """ 
    Retrieve the closest node in semantic meaning from the Graph
    - Similarity Threshold added
    """
    answer_embedding = get_embeddings([answer])

    # Find the most semantically similar node to the answer
    max_similarity = 0
    relevant_node = None

    for node in graph.nodes():
        node_embedding = get_embeddings([node])
        similarity = cosine_similarity(answer_embedding, node_embedding)[0][0]
        
        if similarity > max_similarity and similarity >= threshold:
            max_similarity = similarity
            relevant_node = node

    return relevant_node


# Function to check if two edges are semantically similar
def are_edges_similar(edge1, edge2, threshold=0.8):
    edge1_embedding = get_embeddings([edge1])[0]
    edge2_embedding = get_embeddings([edge2])[0]
    similarity = cosine_similarity([edge1_embedding], [edge2_embedding])[0][0]
    return similarity >= threshold


contradiction_dict = {("Elephant", "should not be discussed by", "You"): "Mentioning Elephant violates the principle"} # Accuracy on parsing contradiction is not good enough here ... 


# Function to check for contradiction using LLM
def check_contradiction_with_llm(node1, edge1, node2, edge2, connected_node1, connected_node2):
    """ 
    Check contradiction between two edges on the graph
    """

    prompt = f"""Compare the following two statements and determine if there's a contradiction:

    1. {node1} {edge1} {connected_node1}
    2. {node2} {edge2} {connected_node2}

    

    Is there a contradiction between these statements? Respond with 'Yes' if there's a contradiction, or 'No' if there isn't. Explain your reasoning briefly.

    Answer:"""

    return call_claude_api(prompt)


def check_contradiction_with_llm_general(node1, edge1, connected_node1, statement):
    """ 
    Check general contradiction between a statement and an edge on the graph
    """
    prompt = f"""Compare the following two statements and determine if there's a contradiction:

    1. {node1} {edge1} {connected_node1}
    2. {statement}

    Is there a contradiction between these statements? Respond with 'Yes' if there's a contradiction, or 'No' if there isn't. Explain your reasoning briefly.

    Answer:"""

    return call_claude_api(prompt)


# Main function to check for contradictions
def check_contradiction(arg_graph, enhanced_graph, node_threshold=0.8, edge_threshold=0.8):
    """ 
    Check Contradiction in two graph: 
    - Iterate through nodes on arg_graph, retrieve semantically similar nodes on enhanced_graph
    - Iterate though neighbours of the node in arg_graph & enhanced_graph, look for contradictions
    - Report contradictions 
    """
    contradictions = []
    
    for node in arg_graph.nodes():
        # Find semantically similar nodes in enhanced_graph
        similar_node = retrieve_node(node, enhanced_graph)
        
        # Check edges of both nodes
        arg_edges = list(arg_graph.edges(node, data=True))
        enhanced_edges = list(enhanced_graph.edges(similar_node, data=True))
        
        # Look for contradiction with similar edges
        for arg_edge in arg_edges:
            for enhanced_edge in enhanced_edges:
                contradiction = check_contradiction_with_llm(
                    node, arg_edge[2]["label"], similar_node, enhanced_edge[2]["label"],
                    arg_edge[1], enhanced_edge[1]
                )
                if contradiction.lower().startswith("yes"):
                    print("Contradiction Found: ", arg_edge[2]["label"], enhanced_edge[2]["label"])
                    print("Explanation: ", contradiction)
                    print()
    
    return contradictions


# Main function to check for contradictions
def check_contradiction(arg_graph, enhanced_graph, node_threshold=0.8, edge_threshold=0.8):
    """ 
    Check Contradiction in two graph: 
    - Iterate through nodes on arg_graph, retrieve semantically similar nodes on enhanced_graph
    - Iterate though neighbours of the node in arg_graph & enhanced_graph, look for contradictions
    - Report contradictions 
    """
    contradictions = []
    
    for node in arg_graph.nodes():
        # Find semantically similar nodes in enhanced_graph
        similar_node = retrieve_node(node, enhanced_graph)
        
        # Check edges of both nodes
        arg_edges = list(arg_graph.edges(node, data=True))
        enhanced_edges = list(enhanced_graph.edges(similar_node, data=True))
        
        # Look for contradiction with similar edges
        for arg_edge in arg_edges:
            for enhanced_edge in enhanced_edges:
                contradiction = check_contradiction_with_llm(
                    node, arg_edge[2]["label"], similar_node, enhanced_edge[2]["label"],
                    arg_edge[1], enhanced_edge[1]
                )
                if contradiction.lower().startswith("yes"):
                    print("Contradiction Found: ", arg_edge[2]["label"], enhanced_edge[2]["label"])
                    print("Explanation: ", contradiction)
                    contradictions.append({
                        "arg_edge": arg_edge[2]["label"],
                        "enhanced_edge": enhanced_edge[2]["label"],
                        "explanation": contradiction
                    })
    
    return contradictions


def check_contradiction(arg_graph, enhanced_graph, node_threshold=0.8, edge_threshold=0.8):
    """ 
    Check Contradiction in two graph: 
    - Iterate through nodes on arg_graph, retrieve semantically similar nodes on enhanced_graph
    - Iterate though neighbours of the node in arg_graph & enhanced_graph, look for contradictions
    - Report contradictions 
    """
    contradictions = []
    
    for node in arg_graph.nodes():
        # Find semantically similar nodes in enhanced_graph
        similar_node = retrieve_node(node, enhanced_graph)
        
        # Check edges of both nodes
        arg_edges = list(arg_graph.edges(node, data=True))
        enhanced_edges = list(enhanced_graph.edges(similar_node, data=True))
        
        # Look for contradiction with similar edges
        for arg_edge in arg_edges:
            for enhanced_edge in enhanced_edges:
                contradiction = check_contradiction_with_llm(
                    node, arg_edge[2]["label"], similar_node, enhanced_edge[2]["label"],
                    arg_edge[1], enhanced_edge[1]
                )
                if contradiction.lower().startswith("yes"):
                    print("Contradiction Found: ", arg_edge[2]["label"], enhanced_edge[2]["label"])
                    print("Explanation: ", contradiction)
                    print()
    
    return contradictions


# Function to generate a query for checking argument adherence
def generate_adherence_query(node, target_node, edge):
    prompt = f"""
    How can I check whether an argument follows this statement:
    {node} {edge} {target_node}

    Please phrase your response as a concise question that could be used to retrieve a relevant function.
    """
    response = call_claude_api(prompt)
    return response.strip()


def get_edge_from_nodes(source, target, llm_model):
    """ 
    Use LLM to infer about the relationship between two nodes, provide the edge's value
    - Provide none if the relationship is not clear
    """
    prompt = f"""Given two nodes in a knowledge graph, '{source}' and '{target}', infer a plausible relationship between them. The relationship should be directional, going from '{source}' to '{target}'. If no clear relationship exists, indicate this.

    Respond in the following JSON format:
    {{
        "edge": "inferred relationship or null if no clear relationship",
        "explanation": "brief explanation of the relationship or why no clear relationship exists"
    }}

    Examples:
    {{
        "edge": "is a type of",
        "explanation": "This relationship indicates that the source is a specific instance or subclass of the target."
    }}

    {{
        "edge": null,
        "explanation": "There is no clear or direct relationship between these two concepts."
    }}

    Now, infer the relationship between: {source} and {target}
    """
    
    response = llm_model.generate(prompt)
    
    try:
        json_start = response.index('{')
        parsed_response = json.loads(response[json_start:])
        inferred_edge = parsed_response['edge']
        return inferred_edge
    except:
        print(f"Failed to infer relationship between {source} and {target}")
        return None
    
    
    
def connect_graph(graph, llm_model):
    """ 
    When graph is separated into two disconnected components, use LLM to infer the relationship between them
    - Base on the most similar nodes, draw a edge to connect the two components
    """
    # Check if the graph is connected
    if nx.number_weakly_connected_components(graph) == 1:
        return graph  # Graph is already connected

    # Get the connected components
    components = list(nx.weakly_connected_components(graph))
    
    if len(components) > 2:
        print("Warning: More than two disconnected components. Only connecting the two largest.")
        components = sorted(components, key=len, reverse=True)[:2]
    
    # Get embeddings for nodes in each component
    embeddings1 = get_embeddings(list(components[0]))
    embeddings2 = get_embeddings(list(components[1]))
    
    # Calculate cosine similarity between all pairs of nodes from different components
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    
    # Find the pair of nodes with highest similarity
    max_i, max_j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
    # For similarity smaller than 0.5, do not connect
    if similarity_matrix[max_i][max_j] < 0.5: # When graph is disconnected
        return graph
        # # Return only the component that includes the start node
        # start_node = list(components[0])[0]  # Choose an arbitrary node from the first component
        # connected_component = nx.weakly_connected_component_subgraphs(graph, copy=False)
        # for component in connected_component:
        #     if start_node in component:
        #         return component
        # # return graph
    
    node1 = list(components[0])[max_i]
    node2 = list(components[1])[max_j]
    
    # Use LLM to infer the relationship between the most similar nodes
    edge = get_edge_from_nodes(node1, node2, llm_model)
    
    if edge:
        # Add the edge to the graph
        graph.add_edge(node1, node2, label=edge)
        print(f"Connected '{node1}' to '{node2}' with relationship: '{edge}'")
    else:
        print(f"Could not infer a relationship between '{node1}' and '{node2}'")
    
    return graph


def get_related_nodes(graph, node):
    """ 
    Get the (start, edge, target) tuples where start = node in the graph
    """
    edges = graph.edges(data=True)
    related_nodes = [(node, edge['label'], target) for source, target, edge in edges if source == node]
    return related_nodes


def retrieve_graph(query, enhanced_graph):
    """ 
    Textual RAG on the logical Graph
    """
    (node, similarity) = retrieve_most_similar_nodes(query, enhanced_graph)
    relations = get_related_nodes(enhanced_graph, node)

    # Verbalize the relations
    # print(f"Most similar node: {node}")
    # print("Relations:")
    relation_strs = []
    for source, relationship, target in relations[::-1]:
        relation_strs.append(f"- {target} {relationship} {source}")
    return ("\n").join(relation_strs)


def store_graph(folder_dir, file_name, graph):
    """ 
    Store the graph into a file, also plot the graph and store the image format 
    """

    # Ensure the directory exists
    os.makedirs(folder_dir, exist_ok=True)

    # Store the graph as a JSON file
    graph_data = nx.node_link_data(graph)
    json_file_path = os.path.join(folder_dir, f"{file_name}.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(graph_data, json_file)

    # Use the visualize_graph function to get the figure
    fig = visualize_graph(graph, True)
    
    image_file_path = os.path.join(folder_dir, f"{file_name}.png")
    fig.savefig(image_file_path)
    plt.close(fig)

    print(f"Graph stored as JSON: {json_file_path}")
    print(f"Graph image saved as: {image_file_path}")
    

def load_graph(folder_dir, file_name):
    """ 
    Load the graph from file into a networkx graph object 
    """

    # Construct the full path to the JSON file
    json_file_path = os.path.join(folder_dir, f"{file_name}.json")

    # Check if the file exists
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"Graph file not found: {json_file_path}")

    # Load the JSON data from the file
    with open(json_file_path, 'r') as json_file:
        graph_data = json.load(json_file)

    # Create a new DiGraph object from the loaded data
    graph = nx.node_link_graph(graph_data, directed=True)

    print(f"Graph loaded from: {json_file_path}")
    return graph

def remove_empty_edge(graph):
    """ 
    Remove empty edge on graph --> Clean Graph
    """
    # Obtain clean graph by removing edges without labels
    clean_graph = nx.DiGraph()

    for u, v, d in graph.edges(data=True):
        if 'label' in d:
            clean_graph.add_edge(u, v, label=d['label'])
    return clean_graph

def preprocess_image(feedback, query, graph):
    """ 
    - Store query-specific heuristic graph under folder specific to the feedback
    - Load Image to base64 | Same for GPT-4o & Claude at least
    """
    # Remove empty edges (with no label)
    graph = remove_empty_edge(graph)
    # Store
    img_folder = f"database/{feedback.file_name}/images"
    os.makedirs(img_folder, exist_ok=True)
    store_graph(img_folder+"/", query.replace(" ", "-"), graph) 
    # Load Image
    image = Image.open(f"{img_folder}/{query.replace(' ', '-')}.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_media_type = "image/png"
    return image_base64, image_media_type


def preprocess_knowledge_graph(graph):
    """ 
    - Store query-specific heuristic graph under folder specific to the feedback
    - Load Image to base64
    """
    # Remove empty edges (with no label)
    graph = remove_empty_edge(graph)
    # Store
    img_folder = f"database/knowledge_graph/images"
    os.makedirs(img_folder, exist_ok=True)
    store_graph(img_folder+"/", "knowledge_graph", graph) 
    # Load Image
    image = Image.open(f"{img_folder}/knowledge_graph.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_media_type = "image/png"
    return image_base64, image_media_type


def preprocess_reverse_image(query, graph):
    """ 
    - Store query-specific heuristic graph under folder specific to the feedback
    - Load Image to base64
    """
    # Remove empty edges (with no label)
    graph = remove_empty_edge(graph)
    # Store
    img_folder = f"database/reverse_relation/images"
    os.makedirs(img_folder, exist_ok=True)
    store_graph(img_folder+"/", query.replace(" ", "-"), graph) 
    # Load Image
    image = Image.open(f"{img_folder}/{query.replace(' ', '-')}.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    image_media_type = "image/png"
    return image_base64, image_media_type


def get_claude_response(query, img = None, img_type = None, system_prompt = "You are a helpful assistant."):
    """ 
    Claude response with query and image input
    """
    # client = anthropic.Anthropic()
    if img is not None:
        text_content = [{"type": "text", "text": query}]
        img_content = [{"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img}}]
    else:
        text_content = query
        img_content = ""
        
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": img_content + text_content,
            }
        ],
        system=system_prompt,
    )
    return message.content[0].text 

##################################
# Make the Language More Natural #
##################################

# I naturally try to rephrase it to be more concise request
def parse_summary(summary):
    """Parse the summarized request from Claude's response."""
    # Remove any potential prefixes like "Summarized request:" or "Here's a concise summary:"
    cleaned = summary.split(":")[-1].strip()
    return cleaned

# Rephrase helps making the request much more natural (works for few instructions case I suppose)
def get_request(feedback, query, model="claude"):
    """ 
    Just to make the joint request more human-readable, therefore more likely to have appeared in the training corpus
    """
    summarize_prompt = f"Summarize the following request in a single concise sentence, starting with 'Summarized request:': \nAnswer the query '{query}' while following the instruction '{feedback.content}'"
    if model == "claude":
        summary_response = call_claude_api(summarize_prompt)
    elif model == "gpt":
        summary_response = get_oai_response(summarize_prompt)
    try:
        summarized_request = parse_summary(summary_response)
        return summarized_request
    except:
        raw_request = f"Provide you response to the query: {query} while adhering to the instruction: {feedback.content}"
        return raw_request 
    
def visual_prompt(feedback, query, graph, advice_str, model="claude", roleplay=False):
    """ 
    - Visual Prompting on Claude Sonnet 3.5
    - Work better with extra summarize gadget
    """
    if roleplay: # Under Roleplay scenario, system prompt & naive query works better | other combination reduces the self awareness of the model
        img, img_type = preprocess_image(feedback, query, graph)
        
        reject = check_reject_in_edges(graph)
        if reject:
            prompt = f"Reject the question, for example 'Sorry I don't xxx, I am just xxx'. Question: {query}"
            # prompt = f"Given your character, question {query} is weird and you will reject answering it. Provide your response: "
        else:
            prompt = f"Answer the question. Question: {query}"
        if model == "claude":
            hg_response = get_claude_response(prompt, img, img_type, system_prompt=feedback.content)
        elif model == "gpt":
            hg_response = get_oai_response(prompt, feedback.content, img, img_type)
        return hg_response.replace("image", "instruction").replace("diagram", "instruction")
    else:
        request = get_request(feedback, query, model=model)
        prompt = f"Follow the instruction on the image and answer the following query: {request}. Do not mention the image and directly provide your answer. Your Answer:"
        img, img_type = preprocess_image(feedback, query, graph)
        if model == "claude":
            hg_response = get_claude_response(prompt, img, img_type)
        elif model == "gpt":
            hg_response = get_oai_response(prompt, img, img_type)
        return hg_response.replace("image", "instruction").replace("diagram", "instruction")

def self_evaluation(instruction, response, model="claude"):
    """ 
    Using sonnet 3.5 to check whether response follows instruction
    """
    prompt = f"Is the following response following the instruction '{instruction}'?\n\nResponse: {response}\n\nAnswer with 'Yes' if it follows the instruction, or 'No' if it doesn't."
    if model == "claude":
        evaluation_result = call_claude_api(prompt)
    else:
        evaluation_result = get_oai_response(prompt, "You are a helpful assistant.")
    follows_instruction = evaluation_result.lower().strip().startswith('yes')
    return follows_instruction


def heuristic_graph_prompting_v0(feedback, query):
    """ 
    Provide response following instruction using Heuristic Graph Prompting
    - Self-Evaluation
    - LLM-aided Graph Traversal & Navigation
    - Visual Prompting on LLM
    """
    instruction = feedback.content
    sys_prompt = f"Instruction: {instruction}"
    response = get_response(query, sys_prompt)
    follow_instruction = self_evaluation(instruction, response)
    if follow_instruction:
        print("Correct Answer Encoutered")
    else:
        print("Incorrect Answer Encountered")
        print("Initializing Heuristic Graph")
        
        # Argument containing instruction and response
        arguments = [instruction, response] 
        
        # LLM-aided graph traversal & navigation
        llm_model = LLMModel(client)
        partial_graph = create_graph_from_arguments(arguments) # Seems to be able to get only one triplet from long-ass argument
        connected_graph = connect_graph(partial_graph, llm_model)
        patched_graph = reverse_graph(connected_graph, llm_model)
        enhanced_graph = enhance_graph(patched_graph, list(patched_graph.nodes())[0], llm_model) # Fixed prompt issue (CoT ver.)
        enhanced_graph = reverse_graph(enhanced_graph, llm_model)
        
        # Visual Prompting on LLM
        fig = visualize_graph(enhanced_graph, return_fig=True)
        hg_response = visual_prompt(feedback, query, enhanced_graph)
        
        if self_evaluation(instruction, hg_response):
            response = hg_response
        else:
            hg_response = visual_prompt(feedback, query, enhanced_graph) # Regenerate visual prompted response 
            if self_evaluation(instruction, response):
                response = hg_response 
            else:
                print("Require human supervision")
                response = FLAG_FOR_SUPERVISION
            
    return response    

#################
# Logical Graph #
#################
# End-to-End Parsing of Logical Graph
# Much like a visualized version of CoT reasoning process (Atomic one, perhaps)

import json
import re

def remove_comment(response):
    # remove lines satisfies line_str.strip().startswith("//"), also remove empty lines
    response_lines = response.split('\n')
    filtered_lines = [line for line in response_lines if not line.strip().startswith("//") and line.strip() != ""]
    filtered_response = '\n'.join(filtered_lines)
    return filtered_response 
    
    
def parse_json_from_response(response):
    """
    Parse out the JSON output from the 'response' string.
    
    Args:
    response (str): The string containing the JSON output.
    
    Returns:
    dict: The parsed JSON data, or None if parsing fails.
    """

    # Find the JSON content using regex
    json_match = re.search(r'\{[\s\S]*\}', response)
    
    if json_match:
        json_str = json_match.group(0)
        json_str = remove_comment(json_str) # OAI special treatment
        advice_str = response.split(json_str)[-1]
        try:
            # Parse the JSON string
            json_data = json.loads(json_str)
            
            # Ensure the JSON structure is as expected
            if 'nodes' in json_data and 'edges' in json_data:
                # No need to modify nodes or edges
                return json_data, advice_str
            else:
                print("Unexpected JSON structure.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON content found in the response.")
        return None


def build_graph_from_json(parsed_json):
    """ 
    Construct graph object from parsed json
    - nodes & edges as the key values
    """

    G = nx.DiGraph()

    # Add nodes
    for node in parsed_json['nodes']:
        try:
            G.add_node(node)
        except:
            G.add_node(node['id'], label=node['label'])
            
    # Add edges
    for edge in parsed_json['edges']:
        G.add_edge(edge['from'], edge['to'], label=edge['relationship'])

    return G 

def parse_logical_graph(response):
    """ 
    Parse out the Logical Graph Adopted by the LLM 
    """
    try:
        parsed_json, advice_str = parse_json_from_response(response)
        if parsed_json is None:
            print("Error: parse_json_from_response returned None.")
            return None
        logical_graph = build_graph_from_json(parsed_json)
        return logical_graph, advice_str
    except json.JSONDecodeError:
        print("Error: Unable to parse the logical graph response as JSON.")
    except KeyError as e:
        print(f"Error: Missing key in the JSON structure: {e}")
    except nx.NetworkXError as e:
        print(f"Error: NetworkX error while building graph: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, ""
    
##################
# Simplification #
##################
# Multi-Hop = 1-Hop + 1-Hot + ....
# 1-Hop can be carried out by LLM 
# Multi-Hop can be baked into LLM

def map_feedback_to_graph_prompt(feedback, query):
    
    if feedback.content == "You should not talk about Elephant":
        get_logical_graph_prompt = f"""Given instruction: {feedback.content}, and query: {query}

        Create a logical graph containing nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.

        For example, given the instruction "Do not talk about elephants" and the query "Discuss the largest land mammal on Earth", one might create this graph:

        {{
            "nodes": ["I", "Elephant", "Largest land mammal on Earth"],
            "edges": [
                {{"from": "I", "to": "Elephant", "relationship": "should not talk about"}},
                {{"from": "Elephant", "to": "Largest land mammal on Earth", "relationship": "is"}},
                {{"from": "I", "to": "Largest land mammal on Earth", "relationship": "should discuss indirectly"}}
            ]
        }}

        Please provide a similar JSON structure for the given instruction and query."""
        
    if feedback.content == "You should roleplay as a customer":
        get_logical_graph_prompt = f"""Given instruction: {feedback.content}, and query: {query}

        Create a logical graph to represent your thinking process about whether you should answer or reject the question based on your roleplay character. Consider the following:

        1. Your role as defined in the instruction
        2. The nature of the query and whether it's appropriate for your character to ask or answer
        
        Provide your response in JSON format with nodes representing key concepts and edges representing relationships between them.

        For example, given the instruction "Roleplay as a customer" and the query "What are your store hours", you might create this graph:

        {{
            "nodes": ["I", "Customer", "Store Hours Question", "Store Employee"],
            "edges": [
                {{"from": "I", "to": "Customer", "relationship": "am roleplaying as"}},
                {{"from": "Customer", "to": "Store Hours Question", "relationship": "can ask"}}
            ]
        }}

        Please create a similar JSON structure for the given instruction and query, ensuring your graph reflects the appropriate reasoning for your character."""
        
    # print("Logical Graph Prompt: ", get_logical_graph_prompt)
    return get_logical_graph_prompt


def map_feedback_to_enhance_prompt(feedback, query):
    if feedback.content == "You should roleplay as a customer":
        enhance_logical_graph_prompt = f"""Given instruction: {feedback.content}, query: {query}

        Analyze the diagram and determine if your roleplay character should ask or answer the question. If your character typically asks such a question, you should REJECT it. If your character typically answers it, you should ANSWER it.
        
        Create a logical graph to represent your decision-making process, focusing on whether to ANSWER or REJECT the Current Question.

        For example, given the instruction "Roleplay as a customer" and the query "What are your store hours", you might enhance the graph as follows:
        {{
            "nodes": ["I", "Customer", "Current Question", "Store Employee"],
            "edges": [
                {{"from": "I", "to": "Customer", "relationship": "am roleplaying as"}},
                {{"from": "Customer", "to": "Current Question", "relationship": "typically asks"}},
                {{"from": "I", "to": "Current Question", "relationship": "should REJECT"}},
                {{"from": "Current Question", "to": "I", "relationship": "should be rejected by"}}
            ]
        }}
        Note that you should keep as less egdes as possible besides the REJECT and ANSWER edges.

        Please provide a similar JSON structure for the given instruction and query. Include reverse relationships and any relevant transitive connections to fully represent the reasoning process."""
    
    elif feedback.content == "You should not talk about Elephant":
        enhance_logical_graph_prompt = f"""Given instruction: {feedback.content}, you are asked: {query}

        Enhance the logical graph by adding reverse relation connections and transitive connections. The graph should contain nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.

        For example, given the instruction "Roleplay as a customer" and the question "What are your store hours", you should realize that the question is for a salesperson, not a customer, an enhanced graph might look like this:

        {{
            "nodes": ["I", "Elephant", "Largest land mammal on Earth"],
            "edges": [
                {{"from": "I", "to": "Elephant", "relationship": "should not talk about"}},
                {{"from": "Elephant", "to": "I", "relationship": "should not be talked about by"}},
                {{"from": "Elephant", "to": "Largest land mammal on Earth", "relationship": "is"}},
                {{"from": "Largest land mammal on Earth", "to": "Elephant", "relationship": "is exemplified by"}},
                {{"from": "I", "to": "Largest land mammal on Earth", "relationship": "should discuss indirectly"}},
                {{"from": "Largest land mammal on Earth", "to": "I", "relationship": "should be discussed indirectly by"}},
                {{"from": "I", "to": "Largest land mammal on Earth", "relationship": "should not talk about example of"}}
            ]
        }}

        Please provide a similar JSON structure for the given instruction and query, ensuring to include reverse relations and transitive connections where appropriate."""

    # print("Enhance Logical Graph Prompt: ", enhance_logical_graph_prompt)
    return enhance_logical_graph_prompt

def get_logical_graph(feedback, query, model="claude", roleplay=False):
    """ 
    Under instruction, directly ask for CoT logical graph
    """
    get_logical_graph_prompt = map_feedback_to_graph_prompt(feedback, query)

    if model == "claude":
        if roleplay:
            txt = call_claude_api(get_logical_graph_prompt, system_prompt=feedback.content)
        else:
            txt = call_claude_api(get_logical_graph_prompt)
    elif model == "gpt":
        if roleplay:
            txt = get_oai_response(get_logical_graph_prompt, system_prompt=feedback.content)
        else:
            txt = get_oai_response(get_logical_graph_prompt)

    print("Test before parsing: ", txt)
        
    logical_graph, advice_str = parse_logical_graph(txt) # bug here

    return logical_graph

def parse_knowledge_graph(txt):
    query = f"""Given the following text: {txt}, create a logical graph containing nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.
    
    For exapmle, given the text 'Table is in the living room.' one might create this graph
    
    {{
    "nodes": ["table", "living room"],
    "edges": [
        {{"from": "table", "to": "living room", "relationship": "is in"}}
    ]
    }}
    
    Please provide a similar JSON structure for the given query."""
    
    txt = call_claude_api(query)
    logical_graph = parse_logical_graph(txt)
    return logical_graph


def reverse_knowledge_graph(txt, graph, reverse=False, compose=False, enhance=False):
    """ 
    Enhace Knowledge Graph with more nodes & edges, together with potential reverse relationship / compositional relationship 
    """
    enhance_prompt = reverse_kg_prompt.format(txt=txt)
    
    img, img_type = preprocess_knowledge_graph(graph)
    
    try:
        response = get_claude_response(enhance_prompt, img, img_type)
        enhanced_knowledge_graph = parse_knowledge_graph(response)
        if enhanced_knowledge_graph:
            return enhanced_knowledge_graph
        else:
            return graph 
    except Exception as e:
        print(f"Error enhancing Knowledge Graph: {str(e)}")
        return graph  # Return original graph when an error occurs

def enhance_knowledge_graph(txt, graph):
    """ 
    Enhace Knowledge Graph with more nodes & edges, together with potential reverse relationship / compositional relationship 
    """
    enhance_kg_prompt = f"""Given the following text: {txt}
    
    Enhance the graph by adding edges of potential reverse relationship / compositional relationship. Provide your response in JSON format.
    
    Reversal Relation Inference:
    For example, given the text 'Einstein received 1921 Physics Nobel Prize', we can infer a reverse relationship:
    - Einstein received 1921 Physics Nobel Prize
    - 1921 Physics Nobel Prize was awarded to Einstein

    Compositional Relation Inference:
    For example, given the additional information 'The Nobel Prize is awarded in Stockholm', we can infer a compositional relationship:
    - Einstein received 1921 Physics Nobel Prize
    - The Nobel Prize is awarded in Stockholm
    - Therefore, Einstein received an award in Stockholm

    Here's how these relationships would be represented in the graph:

    {{
        "nodes": ["Einstein", "1921 Physics Nobel Prize", "Stockholm"],
        "edges": [
            {{"from": "Einstein", "to": "1921 Physics Nobel Prize", "relationship": "received"}},
            {{"from": "1921 Physics Nobel Prize", "to": "Einstein", "relationship": "awarded to"}},
            {{"from": "1921 Physics Nobel Prize", "to": "Stockholm", "relationship": "awarded in"}},
            {{"from": "Einstein", "to": "Stockholm", "relationship": "received award in"}}
        ]
    }}
    
    Please provide a similar JSON structure for the given text, ensuring to include reverse relations and transitive connections where appropriate."""
    
    img, img_type = preprocess_knowledge_graph(graph)
    
    try:
        response = get_claude_response(enhance_kg_prompt, img, img_type)
        enhanced_knowledge_graph = parse_knowledge_graph(response)
        if enhanced_knowledge_graph:
            return enhanced_knowledge_graph
        else:
            return graph 
    except Exception as e:
        print(f"Error enhancing Knowledge Graph: {str(e)}")
        return graph  # Return original graph when an error occurs
        
    

def forward_graph(query):
    # Go the otherway -- given a known knowledge, we explicitly reverse it
    query = f"""Given the query: {query}, create a logical graph containing nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.

    Here's an example:
    Query: Who is the mother of Mary?
    Response:
    {{
    "nodes": ["Mary", "Unknown"],
    "edges": [
        {{"from": "Mary", "to": "Unknown", "relationship": "Is the mother of"}},
        {{"from": "Unknown", "to": "Mary", "relationship": "Is the son of"}}
    ]
    }}
    
    Please follow this format for your response."""
    txt = call_claude_api(query)
    g = parse_logical_graph(txt)
    return g


def reverse_graph(query):
    # Imagination with Unknown Entity here
    # Hope is to elicit better association with the 'imaginative' reverse argument
    query = f"""Given the query: {query}, create a logical graph containing nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.

    Here's an example:
    Query: Who is the son of Mary?
    Response:
    {{
    "nodes": ["Mary", "Unknown"],
    "edges": [
        {{"from": "Mary", "to": "Unknown", "relationship": "Is the mother of"}},
        {{"from": "Unknown", "to": "Mary", "relationship": "Is the son of"}}
    ]
    }}

    Please follow this format for your response."""

    txt = call_claude_api(query)
    g = parse_logical_graph(txt)
    return g


def get_logical_graph_query(query):
    """ 
    Under instruction, directly ask for CoT logical graph
    """
    query = f"Given the query: {query}, create a logical graph containing nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format."
    txt = call_claude_api(query)
    logical_graph = parse_logical_graph(txt)
    return logical_graph

# Customize logical graph prompt for each scenario




def get_logical_graph_with_hint(feedback, query, hint, model="claude", roleplay=False):
    """ 
    Under instruction, directly ask for CoT logical graph
    - hint is the human provided advice to help address the query under instruction-following constraint
    """
    
    get_logical_graph_prompt = map_feedback_to_graph_prompt(feedback, query, hint)

    if model == "claude":
        if roleplay:
            txt = call_claude_api(get_logical_graph_prompt, system_prompt=feedback.content)
        else:
            txt = call_claude_api(get_logical_graph_prompt)
    elif model == "gpt":
        if roleplay:
            txt = get_oai_response(get_logical_graph_prompt, system_prompt=feedback.content)
        else:
            txt = get_oai_response(get_logical_graph_prompt)
            
    logical_graph, advice_str = parse_logical_graph(txt)
    
    return logical_graph, advice_str 


def enhance_logical_graph(feedback, query, graph, model="claude", roleplay=False):
    """ 
    Base on the logical graph, feedback, query, enhance the logical graph by including reverse connection, transitive connection, etc.
    """
    enhance_logical_graph_prompt = map_feedback_to_enhance_prompt(feedback, query)
    
    img, img_type = preprocess_image(feedback, query, graph)
    try:
        if model == "claude":
            if roleplay:
                response = get_claude_response(enhance_logical_graph_prompt, img, img_type, system_prompt=feedback.content)
                print("Response: ", response)
            else:
                response = get_claude_response(enhance_logical_graph_prompt, img, img_type)
        elif model == "gpt":
            if roleplay:
                response = get_oai_response(enhance_logical_graph_prompt, img, img_type, system_prompt=feedback.content)
            else:
                response = get_oai_response(enhance_logical_graph_prompt, img, img_type)
                
        enhanced_logical_graph, advice_str = parse_logical_graph(response)
        if enhanced_logical_graph:
            return enhanced_logical_graph, advice_str
        else:
            return graph, ""  # Return original graph if enhancement fails
    except Exception as e:
        print(f"Error enhancing logical graph: {str(e)}")
        return graph, ""  # Return original graph when an error occurs


def enhance_logical_graph_lossless(feedback, query, graph, model="claude"):
    """ 
    Based on the logical graph, feedback, query, enhance the logical graph by including reverse connection, transitive connection, etc.
    - Keep all the previous nodes & edges, only add new nodes and edges but do not delete existing ones
    """
    
    enhance_logical_graph_prompt = map_feedback_to_enhance_prompt(feedback, query)

    img, img_type = preprocess_image(feedback, query, graph)
    
    try:
        if model == "claude":
            response = get_claude_response(enhance_logical_graph_prompt, img, img_type)
        elif model == "gpt":
            response = get_oai_response(enhance_logical_graph_prompt, img, img_type) # Need to add img & img_type input with OpenAI API.
        
        enhanced_logical_graph, advice_str = parse_logical_graph(response) # OAI response requires extra care
        
        if enhanced_logical_graph:
            # Merge the original graph with the enhanced graph
            merged_graph = graph.copy()
            merged_graph.add_nodes_from(enhanced_logical_graph.nodes())
            merged_graph.add_edges_from([edge for edge in enhanced_logical_graph.edges() if edge not in graph.edges()])
            return merged_graph
        else:
            return graph  # Return original graph if enhancement fails
    except Exception as e:
        print(f"Error enhancing logical graph: {str(e)}")
        return graph  # Return original graph when an error occurs
    
    
def enhance_reverse_graph_lossless(query, graph):
    """ 
    Enhance Reverse Graph iteratively
    """
    enhance_logical_graph_prompt = f"""Given the query: {query}

    Enhance the logical graph by adding more specific nodes and edges to help identify the unknown entity. The graph should contain nodes and edges, where nodes are nouns or concepts, and edges are relationships between them. Provide your response in JSON format.

    Important: Keep all existing nodes and edges from the original graph. Only add new nodes and edges; do not delete or modify any existing ones.

    For example, given the query "Who is the son of Mary?", an enhanced graph might look like this:

    {{
        "nodes": ["Mary", "Unknown", "Male", "Child", "Person"],
        "edges": [
            {{"from": "Mary", "to": "Unknown", "relationship": "is the mother of"}},
            {{"from": "Unknown", "to": "Mary", "relationship": "is the son of"}},
            {{"from": "Unknown", "to": "Male", "relationship": "is a"}},
        ]
    }}

    Please provide a similar JSON structure for the given query, ensuring to include specific attributes and relationships that could help identify the unknown entity, while preserving all existing nodes and edges from the original graph."""

    g = reverse_graph(query)

    img, img_type = preprocess_reverse_image(query, g)

    response = get_claude_response(enhance_logical_graph_prompt, img, img_type)

    try:
        enhanced_logical_graph = parse_logical_graph(response)
        if enhanced_logical_graph:
            # Merge the original graph with the enhanced graph
            merged_graph = graph.copy()
            merged_graph.add_nodes_from(enhanced_logical_graph.nodes())
            for u, v, data in enhanced_logical_graph.edges(data=True):
                if (u, v) not in graph.edges():
                    merged_graph.add_edge(u, v, label=data.get('label', 'unknown'))
            return merged_graph
        else:
            return graph  # Return original graph if enhancement fails
    except Exception as e:
        print(f"Error enhancing logical graph: {str(e)}")
        return graph  # Return original graph when an error occurs
    
    

def visual_prompt_for_advice(feedback, query, graph):
    """ 
    - Visual Prompting on Claude Sonnet 3.5
    - Rephrase the request & Get visual prompted Advice to how to answer the question
    """
    request = get_request(feedback, query)
    # response_prompt = f"Follow the instruction on the image and answer the following query: {request}."
    advice_prompt = f"Follow the instruction on the image and discuss how to address the following query: {request}"
    img, img_type = preprocess_image(feedback, query, graph)
    hg_advice = get_claude_response(advice_prompt, img, img_type)
    return hg_advice


def heuristic_guided_response(feedback, query, graph):
    """ 
    Use Heuristic Graph to directly generate answer suffers from misleading issue (?)
    - Here we use the Graph to provide advice on how to handle the request
    - Then we provide response based on the advice
    """
    # - Visual prompt for Advice 
    # - Use Advice as Thought to guide the response
    hg_advice = visual_prompt_for_advice(feedback, query, graph)
    query_with_advice = f"Given the instruction: {feedback.content}. Provide your answer to the query: {query}. Here is some advice on how to address the query: {hg_advice}. Your answer:"
    return call_claude_api(query_with_advice)


def iterative_graph_prompting(feedback, query, model="claude", roleplay=False, hint=None):
    """ 
    Iterative heuristic prompting with Graph
    """
    # Iter 1: Get Logical Graph 
    if hint:
        graph = get_logical_graph_with_hint(feedback, query, hint, model)
    else:
        graph = get_logical_graph(feedback, query, model, roleplay)

    # Iter 2: (For roleplay we do one iter only) 
    if not roleplay:
        enhanced_graph, advice_str = enhance_logical_graph_lossless(feedback, query, graph, model)
    else:
        enhanced_graph, advice_str = enhance_logical_graph(feedback, query, graph, model, roleplay)
        
    # Step 3: Visual Prompting | Make use of the advice_str here might be beneficial 
    response = visual_prompt(feedback, query, enhanced_graph, advice_str, model, roleplay)
    
    return response
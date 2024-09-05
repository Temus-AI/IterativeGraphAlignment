reverse_kg_prompt = """Given the following text: {txt}
    
    Enhance the graph by adding edges of potential reverse relationship / compositional relationship. Provide your response in JSON format.
    
    Reversal Relation Inference:
    For example, given the text 'Einstein received 1921 Physics Nobel Prize', we can infer a reverse relationship:
    - Einstein received 1921 Physics Nobel Prize
    - 1921 Physics Nobel Prize was awarded to Einstein

    Here's how these relationships would be represented in the graph:

    {{
        "nodes": ["Einstein", "1921 Physics Nobel Prize", "Stockholm"],
        "edges": [
            {{"from": "Einstein", "to": "1921 Physics Nobel Prize", "relationship": "received"}},
            {{"from": "1921 Physics Nobel Prize", "to": "Einstein", "relationship": "awarded to"}},
        ]
    }}
    
    Please provide a similar JSON structure for the given text, ensuring to include reverse relations and transitive connections where appropriate."""
    
    
compose_kg_prompt = """Given the following text: {txt}
    
    Enhance the graph by adding edges of potential compositional relationship. Provide your response in JSON format.

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
            {{"from": "1921 Physics Nobel Prize", "to": "Stockholm", "relationship": "awarded in"}},
            {{"from": "Einstein", "to": "Stockholm", "relationship": "received award in"}}
        ]
    }}
    
    Please provide a similar JSON structure for the given text, ensuring to include reverse relations and transitive connections where appropriate."""
    
    
reverse_compose_kg_prompt = """Given the following text: {txt}
    
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

    
enhance_kg_prompt = """Given the following text: {txt}
    
    Enhance the graph by adding new nodes and edges base on the text. Provide your response in JSON format.
    
    Enhancement Example:
    Given the text "John is the father of Mary, and Mary lives in New York", we can enhance the graph as follows:
    - John is the father of Mary (original relationship)
    - Mary is the daughter of John (reverse relationship)
    - Mary lives in New York (new relationship)
    - John has a child living in New York (compositional relationship)

    {{
        "nodes": ["John", "Mary", "New York"],
        "edges": [
            {{"from": "John", "to": "Mary", "relationship": "is the father of"}},
            {{"from": "Mary", "to": "John", "relationship": "is the daughter of"}},
            {{"from": "Mary", "to": "New York", "relationship": "lives in"}},
            {{"from": "John", "to": "New York", "relationship": "has a child living in"}}
        ]
    }}

    This example demonstrates how to:
    1. Add reverse relationships (father-daughter)
    2. Include new information from the text (living in New York)
    3. Infer compositional relationships (John's connection to New York through Mary)
    
    Please provide a similar JSON structure for the given text, ensuring to include reverse relations and transitive connections where appropriate."""
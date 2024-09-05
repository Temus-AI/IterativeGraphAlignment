# CodeBase for the reasoning agent to retrieve & use
# Each function forms a tool that LLM can use to solve its evaluation problem
# This is also part of the external knowledge base which LLM possesses.
# Much like voyager, this is probably the most reliable way of doing it. Internalizing these functions into LLM would be the next step here.
from .graph import generate_adherence_query

def retrieve_relevant_function(node, edge, target_node):
    # Generate the query
    query = generate_adherence_query(node, target_node, edge)
    print(f"Generated query: {query}")

    # Retrieve relevant functions from the dictionary
    retrieve_dict = prepare_retrieve_dictionary()

    # Find the most similar query in the retrieve_dict
    from difflib import get_close_matches
    closest_match = get_close_matches(query, retrieve_dict.keys(), n=1, cutoff=0.6)

    # Determine the relevant function
    if closest_match:
        relevant_function = retrieve_dict[closest_match[0]]
        print(f"Retrieved function: {relevant_function}")
    else:
        print("No matching function found. Using default function.")
        relevant_function = "check_word_in_argument"

    # Return the relevant function name
    return relevant_function


# Execution of function call response
def execute_function_call(response):
    result_str = ""
    for k in response.content:
        if k.type == "tool_use":
            function_name = k.name  # function name
            arguments = k.input  # arguments dictionary
            # Execute the function
            if function_name in globals():
                result = globals()[function_name](**arguments)
    return result




# Retrieval Ready Functions
def prepare_retrieve_dictionary():
    """
    Prepare a retrieve dictionary with questions as keys and function names as values.
    """
    retrieve_dict = {
        "how to check the inclusion of word in argument": "check_word_in_argument",
        "How can I check if a string contains any words from a specific list while ignoring case?": "check_word_in_argument"
    }
    return retrieve_dict


# Tools for LLM to call
external_tools=[
        {
            "name": "check_word_in_argument",
            "description": "Check if a string contains any words from a specific list while ignoring case",
            "input_schema": {
                "type": "object",
                "properties": {
                    "argument": {
                        "type": "string",
                        "description": "The string to search in",
                    },
                    "word": {
                        "type": "string",
                        "description": "The word to search for",
                    }
                },
                "required": ["argument", "word"],
            },
        },
]


# Functions to execute
def check_word_in_argument(argument: str, word: str) -> bool:
    """ 
    Check if the argument contains the word
    - argument: str
    - word: str
    - return: bool
    """
    included =  word.lower() in argument.lower()
    return included
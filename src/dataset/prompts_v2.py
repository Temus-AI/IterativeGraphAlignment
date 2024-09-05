# Functional Prompts & Parsing Functions
GENERATE_PROMPT_TEMPLATE = """Given a instruction: {content}. {Idea} Please generate 100 queries which could test whether you follow the instruction correctly. Note that for customer roleplay instruction, an example query could be "would you like to know more about our product?". [Example] 1. <prompt1> \n 2. <prompt2> [End] Test Cases: """

EXTRAPOLATE_TEMPLATE = """Feedback: {content}
GIve me a few prompt-completion examples which correctly follow this feedback."""

SELF_PROMPT_TEMPLATE  = """Feedback: {content}
{self_prompt}"""

SEARCH_TEMPLATE = """Given Feedback: {content}
My answer to the query: {prompt} is: {icl_complete} 
Provide your judgement on my answer. And provide Advice on how to better follow the feedback.
[Example]
Judgement: The completion contains inaccuracies.
Issue:
Revised Response:
Advice:
[End]
"""

MAKE_SENSE_CHECK_TEMPLATE = """Judgement on a completion is : {judgement} 
Is the completion good? Answer with Yes or No.
Answer:"""

TEACHER_QUERY_TEMPLATE  = """[FEEDBACK] {content} [END] Here are a few examples of prompt completions that correctly adhere to the feedback [EXAMPLE] Prompt: {prompt} Completion: {completion} [END] Provide your completion of the following prompt. Prompt: {prompt} Completion: """

AUGMENT_QUERY_TEMPLATE = """Provide 10 queries that are close to the provided prompt. These queries will be used to test whether I can follow your feedback correctly. [Prompt] {prompt} [End of Prompt] [Feedback] {feedback_content} [End of Feedback] [Example] 1. <prompt1> \n 2. <prompt2> [End] Your queries: """



# Function Call includes Structured Output, which could hurt the performance of the model
def parse_prompt_from_response(response_router: str) -> list:
    remove_patterns = ["<prompt1>", "<prompt2>", "[End]"] + [f"{i}." for i in range(1,101)][::-1]
    skip_patterns = ["Here are", "Sure, here"]
    points = []
    for l in response_router.split("\n"):
        if any(p in l for p in skip_patterns):
            continue
        for p in remove_patterns:
            l = l.replace(p, "")
        if l:
            points.append(l.strip())
    return points

judge_patterns = ["**Judgement**:\n", "*Judgement*:\n", "Judgement:\n", "Judgement:", "Judgment:"]
issue_patterns = ["**Issue**:\n", "*Issue*:\n", "Issue:\n", "**Issue:**", "Issue:"]
revise_patterns = ["**Revised Response**:\n", "Revised Response:\n", "Revised Response:"]
advice_patterns = ["**Advice**:\n", "Advice:\n", "Advice:"]

def parse_pattern(text, patterns):
    for p in patterns:
        if p in text:
            prefix, suffix = text.split(p)
            prefix = prefix.replace("\n","").replace('"',"")
            suffix = suffix.strip()
            return prefix, suffix
    return "", text
        
def refill_prev(prev, curr):
    if prev == "" and curr !="":
        return curr, ""
    return prev, curr

def parse_search_node(search_node):
    _, judgement_suffix = parse_pattern(search_node, judge_patterns)
    judgement, issue_suffix = parse_pattern(judgement_suffix, issue_patterns)
    issue, revise_suffix = parse_pattern(issue_suffix, revise_patterns)
    revision, advice = parse_pattern(revise_suffix, advice_patterns)
    issue, revision = refill_prev(issue, revision)
    judgement, issue = refill_prev(judgement, issue)
    parse_dict = {"Judgement": judgement, "Issue": issue, "Revised Response": revision, "Advice": advice}
    return parse_dict

def parse_make_sense_check(make_sense_response: str) -> str:
    yes_pattern = ["Yes", "yes"]
    # no_pattern = ["No", "no"]
    for y in yes_pattern:
        if y in make_sense_response:
            return True
    return False
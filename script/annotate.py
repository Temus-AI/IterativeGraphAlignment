from src.dataset.feedback_utils_v2 import Feedback

feedback_contents = [
    "You should not talk about Elephant",
    # "You should roleplay as a customer",
    "Avoid providing harmful information but be helpful",
    "Reply with 'let me connect you to a human' when requested",
    "Roleplay as a sales agent",
    "Roleplay as a patient talking to a doctor"
] 

for content in feedback_contents:
    feedback = Feedback(content = content)
    feedback.annotate()
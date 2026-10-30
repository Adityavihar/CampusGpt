from services.llm import generate_answer


answer = generate_answer(
    "What is VIT-AP?",
    "VIT-AP University is an institution offering undergraduate and postgraduate programs."
)

print(answer)
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.rag_chain import llm

CONDENSE_PROMPT = """
Rewrite the question to be standalone using chat history.

Chat History:
{history}

Follow-up Question:
{question}

Standalone Question:
"""

def build_history(messages, max_turns=4):
    history = []
    for m in messages[-max_turns*2:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history.append(f"{role}: {m['content']}")
    return "\n".join(history)

def rewrite_with_history(question, messages):
    history = build_history(messages)

    prompt = PromptTemplate.from_template(CONDENSE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "history": history,
        "question": question
    }).strip()
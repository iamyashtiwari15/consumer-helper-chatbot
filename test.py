from agents.agent_decision import medical_agent_graph  # or wherever it's saved

input_state = {
    "input": "give me a single solution",
    "image": None,
    "image_type": "",
    "input_type": "text",
    "agent_name": "",
    "response": "",
    "involved_agents": [],
    "bypass_guardrails": False,
    "messages": []
}

# Run the LangGraph with your test state
output = medical_agent_graph.invoke(input_state)

print("\n--- Final Output ---")
print("Agent used:", output['agent_name'])
print("Response:\n", output['response'])
print("Involved Agents:", output['involved_agents'])

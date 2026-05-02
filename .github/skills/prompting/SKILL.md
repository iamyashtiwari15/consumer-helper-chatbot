# SKILL: Prompting Techniques

## WHAT IS PROMPT ENGINEERING?
The practice of designing inputs to LLMs to produce reliable, high-quality outputs.
Natural language is the new "programming language" — how you phrase, structure, and
contextualise instructions directly determines output quality and consistency.

Research-backed techniques consistently improve output quality by **20–60%** on benchmarks.
Chain-of-thought alone improves math/logic accuracy by **15–40%**.

---

## THE ANATOMY OF A GOOD PROMPT

Every well-designed prompt has up to 5 components. Not all are always needed.

```
┌─────────────────────────────────────────────────────────┐
│  1. ROLE      → Who the model should be                 │
│  2. CONTEXT   → Background, documents, state            │
│  3. TASK      → What exactly to do                      │
│  4. FORMAT    → How to structure the output             │
│  5. EXAMPLES  → What good output looks like             │
└─────────────────────────────────────────────────────────┘
```

**Five golden rules (research-backed):**
1. Specific beats abstract — explicit instructions = predictable output
2. Structure beats prose — use lists, XML tags, numbered steps
3. Examples beat explanations — show don't tell
4. Positive beats negative — say what TO do, not what NOT to do
5. Short beats long (when content is equal) — don't pad prompts

---

## TIER 1 — FOUNDATIONAL TECHNIQUES

### 1. Zero-Shot Prompting
Give the model a task with no examples. Relies entirely on pretrained knowledge.

**When to use:** Simple, well-defined tasks; quick prototyping; general questions.

```
Classify the sentiment of this review as Positive, Negative, or Neutral:
"The product works great but shipping took two weeks."
```

**Zero-shot CoT (chain-of-thought):** Add "Let's think step by step" or
"Think through this carefully before answering." This alone unlocks reasoning
without any examples. Very effective on modern models (2024+).

```
A train travels at 80 km/h for 2.5 hours. How far does it go?
Think step by step.
```

---

### 2. Few-Shot Prompting
Provide 2–5 examples of input → output before the actual query.
The model learns the pattern, tone, and format from examples.

**When to use:** Classification, extraction, formatting tasks; when you need
consistent tone; when zero-shot gives inconsistent results.

```
Classify the topic of each sentence:

Sentence: "The Fed raised interest rates by 25 basis points."
Topic: Finance

Sentence: "The quarterback threw three touchdowns in the final quarter."
Topic: Sports

Sentence: "Scientists discovered a new exoplanet in the habitable zone."
Topic: Science

Sentence: "The prime minister announced an early election."
Topic: Politics
```

**Best practices for few-shot examples:**
- Cover the range of cases (easy + hard + edge cases)
- Keep examples consistent in style and format
- 3–5 examples is usually optimal; beyond 8 brings diminishing returns
- On modern LLMs, few-shot examples primarily align OUTPUT FORMAT —
  not necessarily improve reasoning over zero-shot CoT

---

### 3. Role Prompting (Persona)
Assign the model a specific identity, expertise, or perspective.
Narrows focus, sharpens tone, and produces more expert-level answers.

**When to use:** Almost always — one of the highest-ROI techniques.

```
You are a senior Python engineer specialising in production-grade API design.
Review the following code for security vulnerabilities, performance bottlenecks,
and maintainability issues. Be direct and specific.

[CODE HERE]
```

**Tips:**
- Include domain + seniority level ("senior security researcher" beats "expert")
- Add a behaviour trait if relevant ("be direct", "use simple language", "be concise")
- Combine with format constraints for best results

---

### 4. System Prompt Design
For API/agent usage — the system prompt sets the model's persistent behaviour,
constraints, and persona across the entire conversation.

```python
SYSTEM_PROMPT = """
You are a customer support agent for Acme Corp, a B2B SaaS company.

RULES:
- Only answer questions related to Acme products
- If you don't know the answer, say "I'll escalate this to the team"
- Never discuss competitors by name
- Always end responses with a follow-up question to confirm resolution

TONE: Professional, warm, concise. Max 3 paragraphs per response.
"""
```

**System prompt structure (recommended order):**
1. Identity / role
2. Core task or purpose
3. Rules and constraints (as a numbered list)
4. Tone and format instructions
5. Edge case handling

**Key insight:** Constraints in the system prompt are more reliable than constraints
in the user prompt. Put guardrails in the system prompt.

---

## TIER 2 — REASONING TECHNIQUES

### 5. Chain-of-Thought (CoT) Prompting
Forces the model to show intermediate reasoning steps before the final answer.
Dramatically improves accuracy on multi-step problems.

**When to use:** Math, logic, multi-step reasoning, debugging, analysis.

**Zero-shot CoT:**
```
Q: A store sells 3 types of products. Type A costs $12, Type B costs $18,
Type C costs $25. If a customer buys 4 of A, 2 of B, and 1 of C,
what is the total before a 10% discount?

A: Let's think step by step.
```

**Few-shot CoT (higher reliability):**
```
Q: If there are 5 teams in a league and each plays every other team twice,
how many total games are played?
A: First, count unique pairings: 5 teams → 5×4/2 = 10 unique pairs.
   Each pair plays twice → 10 × 2 = 20 total games.
   Answer: 20

Q: A recipe makes 24 cookies using 3 cups of flour.
How much flour do I need for 40 cookies?
A: [model generates step-by-step reasoning]
```

**CoT variants:**
| Variant | Trigger phrase | Best for |
|---|---|---|
| Zero-shot CoT | "Think step by step" | Quick reasoning boost |
| Verification CoT | "Check your work before answering" | Math, logic |
| Step-back CoT | "First identify the key principles, then solve" | Physics, law |
| Concise CoT | "Reason briefly, then give the answer" | Faster + cheaper |

---

### 6. Self-Consistency Prompting
Run the same prompt multiple times (with temperature > 0), collect all answers,
then pick the most frequent / consistent answer via majority vote.

**When to use:** High-stakes decisions; arithmetic; any task where accuracy
matters more than cost; reduces single-chain reasoning errors.

```python
import anthropic
from collections import Counter

client = anthropic.Anthropic()

def self_consistent_answer(question: str, runs: int = 5) -> str:
    answers = []
    for _ in range(runs):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.7,     # non-zero to get diverse reasoning paths
            messages=[{"role": "user", "content": f"{question}\nThink step by step."}]
        )
        # Extract final answer from reasoning
        answers.append(response.content[0].text.split("Answer:")[-1].strip())
    
    # Return majority vote
    return Counter(answers).most_common(1)[0][0]
```

---

### 7. Tree-of-Thought (ToT) Prompting
Extends CoT from a linear chain to a branching tree.
The model explores multiple reasoning paths simultaneously, evaluates each,
and picks the best branch. Near-human depth of thinking on complex problems.

**When to use:** Creative planning, strategic decisions, complex multi-variable
problems, game-playing tasks, design challenges.

```
Problem: Design a backend architecture for a chat app that must handle
1M concurrent users with sub-100ms latency.

Think about this using a tree of thought:
1. Generate 3 possible high-level approaches
2. For each approach, identify strengths and one critical weakness
3. Eliminate the weakest approach
4. Develop the remaining two approaches one step further
5. Select the best final approach and justify your choice
```

---

### 8. ReAct Prompting (Reason + Act)
The model interleaves **Thought → Action → Observation** in a loop.
Core pattern behind most LLM agents. Combines internal reasoning with
external tool calls (web search, calculators, databases, APIs).

**When to use:** Agentic tasks, anything requiring real-time info, multi-step
tasks with external dependencies. This is what LangGraph agents use under the hood.

```
You have access to: [web_search], [calculator], [database_query]

Task: Find the current USD to INR exchange rate and calculate how much
₹50,000 is in USD.

Thought: I need the current exchange rate, which I don't know. I should search for it.
Action: web_search("USD to INR exchange rate today")
Observation: 1 USD = 83.42 INR as of May 2025

Thought: Now I can calculate. ₹50,000 ÷ 83.42 = ?
Action: calculator(50000 / 83.42)
Observation: 599.16

Thought: I have the answer now.
Final Answer: ₹50,000 = approximately $599.16 USD
```

In LangChain/LangGraph, ReAct is the default agent loop — you don't write this
manually, the framework handles the Think/Act/Observe cycle.

---

## TIER 3 — STRUCTURAL TECHNIQUES

### 9. Prompt Chaining
Break a complex task into sequential prompts where each output feeds the next.
More reliable than one giant prompt. Easier to debug.

**When to use:** Document processing pipelines, multi-stage analysis,
content creation workflows, RAG generation.

```python
# Stage 1: Extract key facts
facts = llm.invoke(f"Extract the 5 most important facts from:\n{document}")

# Stage 2: Identify gaps
gaps = llm.invoke(f"Given these facts:\n{facts}\nWhat critical information is missing?")

# Stage 3: Generate questions
questions = llm.invoke(f"Based on these gaps:\n{gaps}\nGenerate 3 research questions.")

# Stage 4: Synthesize final output
report = llm.invoke(f"Facts: {facts}\nGaps: {gaps}\nQuestions: {questions}\nWrite a 500-word summary.")
```

**Prompt chaining vs. single prompt:**
| | Single prompt | Prompt chaining |
|---|---|---|
| Complexity | Limited by context window | Handles any scale |
| Debugging | Hard to isolate failures | Inspect each stage |
| Cost | One LLM call | Multiple calls |
| Reliability | Degrades on complex tasks | High |

---

### 10. Structured Output Prompting
Force the model to respond in a specific format (JSON, XML, Markdown, table).
Essential for programmatic use — parsing structured output reliably.

**JSON output:**
```
Extract the following from the job posting and return ONLY valid JSON.
No explanation, no markdown, no preamble.

Schema:
{
  "title": "string",
  "company": "string",
  "salary_min": number or null,
  "salary_max": number or null,
  "required_skills": ["string"],
  "experience_years": number or null,
  "remote": boolean
}

Job posting:
[JOB TEXT HERE]
```

**XML output (works well with Claude specifically):**
```
Analyse this customer complaint. Return your analysis in XML:

<analysis>
  <sentiment>positive|negative|neutral</sentiment>
  <category>billing|technical|shipping|other</category>
  <urgency>low|medium|high</urgency>
  <summary>one sentence summary</summary>
  <recommended_action>what support should do</recommended_action>
</analysis>

Complaint: [TEXT]
```

**Using Pydantic for guaranteed structure (LangChain):**
```python
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

class ExtractedJob(BaseModel):
    title: str
    company: str
    salary_min: float | None
    required_skills: list[str]
    remote: bool

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
structured_llm = llm.with_structured_output(ExtractedJob)
result = structured_llm.invoke("Extract from: [JOB TEXT]")
# result is a validated ExtractedJob object — no parsing needed
```

---

### 11. Positive + Negative Examples
Show the model what you want AND what you don't want.
Dramatically reduces unwanted behaviours.

```
Write a product description for noise-cancelling headphones.

GOOD example (do this):
"Immerse yourself in music with 30 hours of battery life and
adaptive noise cancellation that adjusts to your environment."

BAD example (do NOT do this):
"These headphones are really good and have lots of great features
that many customers enjoy and appreciate for their daily use."

Now write the description for: [PRODUCT DETAILS]
```

---

### 12. Meta-Prompting
Ask the model to generate or improve its own prompt.
Useful for bootstrapping and prompt optimisation without manual iteration.

```
I want an LLM to help junior developers write better commit messages.
Generate a detailed system prompt for this use case.
Include: role, rules, 2 examples (good vs bad commit), output format.
```

Or for self-improvement:
```
Here is my current prompt:
---
[YOUR PROMPT]
---
Identify 3 weaknesses in this prompt and rewrite it to fix them.
Explain each change you made.
```

---

## TIER 4 — AGENTIC / ADVANCED TECHNIQUES

### 13. Self-Refinement / Reflection
The model critiques its own output and iteratively improves it.
Adds 10–25% quality improvement over single-pass generation.

```python
def self_refine(task: str, iterations: int = 2) -> str:
    # Initial draft
    draft = llm.invoke(task)
    
    for _ in range(iterations):
        # Critique the draft
        critique = llm.invoke(f"""
        Review this draft for quality, accuracy, and completeness:
        ---
        {draft}
        ---
        List specific improvements needed (be critical):
        """)
        
        # Revise based on critique
        draft = llm.invoke(f"""
        Original task: {task}
        
        Draft: {draft}
        
        Critique: {critique}
        
        Now write an improved version that addresses all critique points:
        """)
    
    return draft
```

---

### 14. Prompt Decomposition (Least-to-Most)
Break a hard problem into sub-problems, solve them in order,
and use each solution to help solve the next. Teaches complex reasoning
through scaffolding.

```
I want to build a RAG system. Let's break this down:

Sub-problem 1: What is a vector database and why do I need one?
[solve this first]

Sub-problem 2: Given the above, which vector database should I choose for
a production system with 1M documents?
[solve using answer to sub-problem 1]

Sub-problem 3: Given the database choice above, write the Python code
to connect and insert documents.
[solve using answer to sub-problem 2]
```

---

### 15. Constitutional Prompting (Self-Critique with Rules)
Give the model a set of "constitution" rules. After generating an output,
the model checks its own output against each rule and revises.
Used by Anthropic's Constitutional AI approach.

```
Generate a response to this user question: [QUESTION]

Then check your response against these rules:
1. Is every factual claim accurate?
2. Does the response avoid speculative language presented as fact?
3. Is the response helpful and complete?
4. Is the tone appropriate for a professional context?

For any rule violation, revise the relevant part.
Return the final, rule-compliant response only.
```

---

## PROMPT TEMPLATES FOR COMMON USE CASES

### RAG / Document QA
```
You are a helpful assistant. Answer questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have that information in the provided documents."
Do not use external knowledge.

Context:
<context>
{retrieved_chunks}
</context>

Question: {user_question}

Answer:
```

### Classification / Routing
```
Classify the following user query into exactly ONE category.
Return ONLY the category name, nothing else.

Categories:
- document: asks about content in uploaded files
- web: needs current or real-world information
- general: small talk, greetings, simple questions

Query: {user_query}
Category:
```

### Code Review
```
You are a senior {language} engineer. Review the following code.

For each issue found, format your response as:
SEVERITY: [critical|major|minor]
LINE: [line number if applicable]
ISSUE: [what is wrong]
FIX: [how to fix it]

Focus on: security, performance, readability, edge cases.
Do NOT comment on style unless it causes bugs.

Code:
{code}
```

### Entity Extraction
```
Extract all entities from the text below.
Return ONLY valid JSON — no markdown, no preamble, no explanation.

{
  "people": ["full names"],
  "organisations": ["company/org names"],
  "locations": ["cities, countries, places"],
  "dates": ["any dates or time references"],
  "amounts": ["monetary values, quantities with units"]
}

Text: {text}
```

---

## TEMPERATURE GUIDE

Temperature controls randomness/creativity vs. determinism.

| Temperature | Output style | Best for |
|---|---|---|
| 0.0 | Fully deterministic | Classification, extraction, code generation |
| 0.1–0.3 | Very consistent | Factual QA, structured outputs, RAG answers |
| 0.4–0.6 | Balanced | Summarisation, explanations, customer support |
| 0.7–0.9 | Creative, varied | Brainstorming, marketing copy, creative writing |
| 1.0+ | Very diverse | Poetry, experimental, diverse options generation |

**Rule:** If you need to parse the output programmatically → use temperature 0.
If you want creative variety → use 0.7–0.9 with self-consistency voting.

---

## DEBUGGING PROMPTS — WHEN OUTPUT IS WRONG

```
Step 1: Isolate → remove half the prompt, see if the problem persists
Step 2: Add CoT → "think step by step" before the answer
Step 3: Add examples → 2–3 demonstrations of correct behaviour
Step 4: Add constraints → "only use information from the context"
Step 5: Check format → are you getting the right structure?
Step 6: Split into chain → break the big prompt into smaller stages
Step 7: Try a stronger model → same prompt, GPT-4o vs GPT-3.5, Claude Sonnet vs Haiku
```

**Common failure modes and fixes:**

| Problem | Likely cause | Fix |
|---|---|---|
| Hallucinating facts | No grounding | Add RAG or "only use provided context" |
| Inconsistent format | No format example | Add a complete example of desired output |
| Too verbose / too short | No length constraint | "Respond in max 3 sentences" / "Write 500 words" |
| Wrong tone | No persona | Add role + tone instruction |
| Reasoning errors | Single pass | Add CoT or self-consistency |
| Ignoring constraints | Constraint in user turn | Move rules to system prompt |
| Mixing tasks | Multi-task prompt | Split into a prompt chain |

---

## QUICK REFERENCE — TECHNIQUE SELECTOR

```
Is the task simple (classification, extraction, QA)?
  → Zero-shot or Few-shot

Does it need step-by-step reasoning (math, logic, analysis)?
  → Chain-of-Thought (zero-shot CoT first, few-shot if needed)

Does accuracy really matter (high-stakes)?
  → Self-Consistency (5+ runs + majority vote)

Does it explore possibilities (design, strategy, creative)?
  → Tree-of-Thought

Does it need external tools or real-time info?
  → ReAct (or LangGraph agent)

Is it a multi-step pipeline?
  → Prompt Chaining

Does output need to be parsed programmatically?
  → Structured Output (JSON/XML + Pydantic)

Is the output mediocre and needs polish?
  → Self-Refinement (critique + revise loop)
```

---

## MODEL-SPECIFIC NOTES (2025)

| Model | Best format | Notes |
|---|---|---|
| Claude (Anthropic) | XML tags | Responds well to `<instructions>`, `<context>`, `<format>` |
| GPT-4o (OpenAI) | Markdown / JSON schema | Structured output API for guaranteed JSON |
| Gemini 1.5 Pro | Natural language | Long context window; good at doc grounding |
| Llama 3 / open-source | Alpaca/Instruct format | Check model card for exact prompt template |

**For Claude specifically:**
```xml
<system>
You are a [ROLE]. Your task is to [TASK].
</system>

<context>
{background_information}
</context>

<instructions>
1. [Step 1]
2. [Step 2]
3. [Step 3]
</instructions>

<format>
Return your answer as JSON: {"key": "value"}
</format>

<question>
{user_query}
</question>
```

---

## KEY DEPENDENCIES / TOOLS

```bash
pip install langchain langchain-anthropic langchain-openai
pip install anthropic openai
pip install pydantic  # for structured outputs
```

**Useful resources:**
- https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- https://www.promptingguide.ai
- LangSmith for prompt versioning and A/B testing in production
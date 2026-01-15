reflection_prompt = """
You have just completed one episode of interaction with the environment.

Your task is to summarize ONLY the information about the environment’s hidden state that was revealed during this episode.
This summary will be used as the persistent belief state for future episodes.

IMPORTANT RULES:
- Do NOT restate the interaction history.
- Do NOT include strategy, planning, or action evaluation.
- Only include information that is directly supported by environment feedback.
- Clearly distinguish confirmed information from remaining uncertainty.
- Be conservative: if something is not guaranteed, mark it as uncertain.
- Do NOT include step-by-step reasoning or internal chain-of-thought.
- Be concise, precise, and structured.

Output your response strictly between the tags:
<summary>
...
</summary>

Inside <summary>, include the following sections:

1. Episode Status
- Whether the hidden state was fully identified or only partially constrained.
- If fully identified, explicitly state the hidden state.

2. Confirmed Information
- Facts about the hidden state that are definitively true.

3. Ruled-Out Possibilities
- States, values, or hypotheses that are now known to be impossible.

4. Remaining Uncertainty
- Aspects of the hidden state that are still unknown.
- Describe the remaining possible space (e.g., ranges, sets, patterns).

5. Carryover Belief
- A concise summary of the agent’s current belief about the hidden state,
  suitable for initializing the next episode.
"""


reflection_prompt_2 = """
[Reflection] Now it’s your turn to reflect on the past experience and come up with a new plan of action.
- Your response should first be step-by-step reasoning about the strategy and path you took
  to attempt to complete the task. Identify where things went wrong or could be better.
- Then devise a concise, new plan of action that accounts for your mistake with reference to
  specific actions that you should have taken.
- Finally, end the response with your reflection and improved plan inside <remark>
  </remark> tags, to guide the next trial.
"""


Hangman_reflection_prompt = """
You have completed one episode of a Hangman game.

Your task is to summarize ONLY the information revealed by the environment during this episode.
This summary will be used as persistent information for future episodes.

Rules:
- Do NOT restate the interaction history.
- Do NOT explain reasoning or strategy.
- Only include facts that are directly supported by the game’s feedback.
- Clearly distinguish confirmed information from remaining uncertainty.
- Be concise and factual.

Output your result strictly between the tags:
<summary>
...
</summary>

Inside <summary>, include:

1. Word Structure
- Word length.
- Known letter positions (use _ for unknown positions).

2. Confirmed Letters
- Letters that are in the word (with positions if known).

3. Excluded Letters
- Letters that are confirmed NOT to appear in the word.

4. Remaining Uncertainty
- Description of which letters and positions are still unknown.

5. Carryover Belief
- A single concise sentence describing the current belief about the target word for the next episode.
"""


summary_prompt = """
You are the EPISODE SUMMARIZER for a meta-RL agent.

Your job now is to **reflect on the past experience**, summarize the information gained from the past experience, 
and provide your summary inside <summary>
  </summary> tags, to guide the next trial.

The history of the past experience is:\n\n
"""


reflection_prompt_3 = """
You have just completed one episode of interaction with the environment.

Your task is to summarize what you have learned from this episode and reflect on how it should influence future episodes.

IMPORTANT RULES:
- Do NOT restate the full interaction history.
- Only include information that is useful for future decision-making.
- Separate confirmed facts from hypotheses or uncertainties.
- Do NOT include step-by-step reasoning or internal chain-of-thought.
- Be concise, precise, and structured.

Please produce the following sections:

1. Episode Outcome
- Did the episode succeed or fail?
- If applicable, what was the final result (e.g., win/loss, score, reward)?

2. Confirmed Information
- List facts that are definitively true based on feedback from the environment.
- These should be safe to rely on in future episodes.

3. Ruled-Out Possibilities
- List actions, states, or hypotheses that are now known to be incorrect or impossible.

4. Uncertain or Probabilistic Beliefs
- Information that is suggested but not guaranteed.
- Clearly mark uncertainty.

5. Strategy Reflection
- What worked well in this episode?
- What was ineffective or misleading?

6. Forward-Looking Guidance
- Concrete advice or constraints for future episodes.
- Focus on improving efficiency, exploration, or accuracy.

Write your response in a compact, bullet-point style.
"""


reflection_prompt_4 = """
You have just completed one episode of interaction with the environment.

Your task is to summarize what you have learned from this episode and reflect on how it should influence future episodes.

IMPORTANT RULES:
- Do NOT restate the full interaction history.
- Only include information that is useful for future decision-making.
- Separate confirmed facts from hypotheses or uncertainties.
- Do NOT include step-by-step reasoning or internal chain-of-thought.
- Be concise, precise, and structured.

Please produce the following sections:

1. Episode Outcome
- Did the episode succeed or fail?
- If applicable, what was the final result (e.g., win/loss, score, reward)?

2. Confirmed Information
- List facts that are definitively true based on feedback from the environment.
- These should be safe to rely on in future episodes.

3. Ruled-Out Possibilities
- List actions, states, or hypotheses that are now known to be incorrect or impossible.

4. Uncertain or Probabilistic Beliefs
- Information that is suggested but not guaranteed.
- Clearly mark uncertainty.

5. Strategy Reflection
- What worked well in this episode?
- What was ineffective or misleading?

6. Forward-Looking Guidance
- Concrete advice or constraints for future episodes.
- Focus on improving efficiency, exploration, or accuracy.

Write your response in a compact, bullet-point style.
"""

"""Domain wording for the native planning/coding pipeline; no search changes."""


def adapt_prompt(agent, prompt, stage):
    task = getattr(agent, "alphaevolve", None)
    if not task:
        return prompt["Introduction"]
    from agents.prompts.impl_guideline import get_impl_guideline_from_agent
    actions = {
        "draft": "Design and implement a first feasible construction algorithm, using the simple initial construction as a starting point when useful.",
        "improve": "Improve the supplied parent algorithm based on its measured score and execution feedback. Implement a concrete algorithmic change.",
        "debug": "Repair the concrete execution or feasibility error in the parent program while preserving its mathematical objective.",
        "evolution": "Use the branch trajectory to design an improved construction algorithm, addressing the observed stagnation.",
        "fusion": "Combine useful algorithmic ideas from the supplied parent and reference solutions, maintaining feasibility.",
        "aggregation": "Combine the strongest complementary ideas from the supplied branch trajectories into a new feasible algorithm.",
    }
    prompt["Introduction"] = "You are an expert in mathematical algorithm discovery and numerical optimization. "+actions[stage]
    # Keep authentic MLEvolve evolution instructions when explicitly selected as a control.
    if stage == "evolution" and getattr(agent.cfg, "audited_bo", "baseline") == "preserved_evolution":
        return prompt["Introduction"]
    prompt["Instructions"] = {
        "Response format": "Return a brief concrete plan followed by one Python code block containing the complete candidate module.",
        "Planning": ["Explain the construction representation, objective and feasibility checks; propose a bounded CPU algorithm.",
                     "Use execution feedback and retrieved memory. For improvements, explain what changes relative to the parent.",
                     "Preserve a feasible incumbent and return it before the deadline; use a small feasibility margin.",
                     "Do not replace the mathematical objective with an ML surrogate, a reference score, or an unverified proxy."],
        **get_impl_guideline_from_agent(agent),
    }
    return prompt["Introduction"]

from __future__ import annotations

from agentz.profiles.base import Profile


# Profile instance for chrome agent
chrome_profile = Profile(
    instructions="You are a chrome agent. Your task is to interact with the chrome browser following the instructions provided.",
    runtime_template="{instructions}",
    output_schema=None,
    tools=None,
    model=None
)

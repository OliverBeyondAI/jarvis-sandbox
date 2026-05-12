"""
Xena Marketing Agent — Agentic AI marketing content generator.

Uses Claude Opus 4.7 and the Anthropic beta managed-agents API to:
  - Research markets, craft brand-aligned messaging, and generate
    multi-channel marketing campaigns autonomously.
  - Generate 7-day social media content plans with captions and
    suggested images, adapting to a specified brand voice.
  - Iteratively refine content via a simulated feedback loop.

Two entry points:
  Campaign mode:  xena_marketing_agent.agent.XenaMarketingAgent
  Social mode:    xena_marketing_agent.social_agent.run_social_agent
"""

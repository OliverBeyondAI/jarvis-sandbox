"""
Dispatch Agent — Parallel research coordination via channels.

Uses Claude Agent SDK Dispatch and Channels patterns to coordinate
multiple research sub-agents for multi-step web research, query
decomposition, and information synthesis.
"""

from .channels import (
    Channel,
    ChannelClosedError,
    ChannelTimeoutError,
    FanOutChannel,
    Mailbox,
    Message,
    MessageType,
    SubAgentResult,
    SubAgentTask,
)
from .dispatcher import Dispatcher, DispatchResult
from .sub_agents import ResearchSubAgent, run_sub_agent_worker

__all__ = [
    "Channel",
    "ChannelClosedError",
    "ChannelTimeoutError",
    "Dispatcher",
    "DispatchResult",
    "FanOutChannel",
    "Mailbox",
    "Message",
    "MessageType",
    "ResearchSubAgent",
    "SubAgentResult",
    "SubAgentTask",
    "run_sub_agent_worker",
]

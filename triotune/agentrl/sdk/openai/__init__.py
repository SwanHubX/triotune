"""OpenAI-compatible tools, sessions, transports, and sampling backends.

This namespace groups the layers of the integration without hiding their
ownership boundaries: ``function_tool`` describes local callable schemas,
``Agent`` runs the tool loop, ``OpenAISession`` records identity and delayed
rewards, ``OpenAIClient`` selects a request backend, and ``OpenAIProxyServer``
offers optional local HTTP forwarding.  ``PyTRIOBackend`` is the training-only
backend that binds a fixed ``SamplingSnapshot``; ordinary provider responses
remain diagnostic API traces rather than PPO trajectories.
"""

from .agent import Agent, AgentRunResult
from .backend import ExternalOpenAIBackend, OpenAIBackend, PyTRIOBackend
from .client import OpenAIClient
from .function_tool import function_tool
from .native import (
    OpenAIChatObservationFormatter,
    OpenAINativeActionParser,
    OpenAIResponsesObservationFormatter,
)
from .proxy_server import OpenAIProxyServer
from .session import OpenAIInteraction, OpenAISession, OpenAITrajectoryExport
from .workflow import OpenAIWorkflow

__all__ = [
    "Agent",
    "AgentRunResult",
    "ExternalOpenAIBackend",
    "OpenAIBackend",
    "OpenAIChatObservationFormatter",
    "OpenAIClient",
    "OpenAIInteraction",
    "OpenAINativeActionParser",
    "OpenAIProxyServer",
    "OpenAIResponsesObservationFormatter",
    "OpenAISession",
    "OpenAITrajectoryExport",
    "OpenAIWorkflow",
    "PyTRIOBackend",
    "function_tool",
]

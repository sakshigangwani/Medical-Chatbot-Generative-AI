"""
FlowSync AI Health Assistant - Orchestrator
Routes user questions to appropriate specialized agents.
"""

from .router import QuestionRouter

__all__ = ['QuestionRouter']

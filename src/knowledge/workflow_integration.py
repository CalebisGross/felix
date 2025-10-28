"""
Workflow Integration Bridge for Knowledge Brain

Integrates knowledge brain with Felix workflows and agents:
- Auto-augmentation: Automatically inject relevant knowledge into workflow context
- Agent extension: Add query_knowledge_brain() method to agents
- Usage tracking: Record which knowledge was useful for workflows

This module bridges the knowledge brain to Felix's existing workflow system
without modifying core Felix code.
"""

import logging
from typing import Optional, Dict, Any, List

from .retrieval import KnowledgeRetriever

logger = logging.getLogger(__name__)


class KnowledgeBrainIntegration:
    """
    Integration bridge between knowledge brain and Felix workflows.

    Provides workflow augmentation and agent extension capabilities.
    """

    def __init__(self, knowledge_retriever: KnowledgeRetriever):
        """
        Initialize knowledge brain integration.

        Args:
            knowledge_retriever: KnowledgeRetriever instance
        """
        self.knowledge_retriever = knowledge_retriever
        self.current_workflow_id = None
        self.current_workflow_knowledge = []

    def augment_workflow_context(self,
                                 task_description: str,
                                 task_type: Optional[str] = None,
                                 task_complexity: Optional[str] = None,
                                 max_concepts: int = 10,
                                 auto_apply: bool = True) -> str:
        """
        Auto-augment workflow with relevant knowledge.

        This method should be called before spawning agents to enrich
        their context with domain knowledge.

        Args:
            task_description: Description of the task
            task_type: Optional task type
            task_complexity: Optional complexity level
            max_concepts: Maximum concepts to inject
            auto_apply: Whether to automatically apply augmentation

        Returns:
            Augmented context string (empty if no relevant knowledge)
        """
        if not auto_apply:
            return ""

        try:
            # Retrieve relevant knowledge
            augmented_context = self.knowledge_retriever.build_augmented_context(
                task_description=task_description,
                task_type=task_type,
                task_complexity=task_complexity,
                max_concepts=max_concepts
            )

            # Track knowledge IDs for later usage recording
            if augmented_context:
                retrieval_result = self.knowledge_retriever.search(
                    query=task_description,
                    task_type=task_type,
                    task_complexity=task_complexity,
                    top_k=max_concepts
                )
                self.current_workflow_knowledge = [r.knowledge_id for r in retrieval_result.results]

            return augmented_context

        except Exception as e:
            logger.error(f"Failed to augment workflow context: {e}")
            return ""

    def record_workflow_outcome(self,
                               workflow_id: str,
                               task_type: str,
                               task_complexity: Optional[str] = None,
                               workflow_success: bool = True,
                               final_confidence: Optional[float] = None):
        """
        Record workflow outcome for meta-learning.

        Should be called after workflow completes to track knowledge utility.

        Args:
            workflow_id: Unique workflow ID
            task_type: Type of task
            task_complexity: Optional complexity level
            workflow_success: Whether workflow succeeded
            final_confidence: Optional confidence score
        """
        if not self.current_workflow_knowledge:
            return  # No knowledge was used

        try:
            # Calculate usefulness score
            # High confidence success = very useful
            # Low confidence success = somewhat useful
            # Failure = not useful
            if workflow_success:
                if final_confidence and final_confidence >= 0.8:
                    useful_score = 0.9
                elif final_confidence and final_confidence >= 0.6:
                    useful_score = 0.7
                else:
                    useful_score = 0.5
            else:
                useful_score = 0.2  # Knowledge didn't help prevent failure

            # Record usage
            self.knowledge_retriever.record_usage(
                workflow_id=workflow_id,
                knowledge_ids=self.current_workflow_knowledge,
                task_type=task_type,
                task_complexity=task_complexity,
                useful_score=useful_score
            )

            logger.info(f"Recorded knowledge usage for workflow {workflow_id}: "
                       f"{len(self.current_workflow_knowledge)} concepts, "
                       f"usefulness={useful_score:.2f}")

            # Clear for next workflow
            self.current_workflow_knowledge = []

        except Exception as e:
            logger.error(f"Failed to record workflow outcome: {e}")

    def extend_agent_with_knowledge_brain(self, agent, knowledge_retriever: Optional[KnowledgeRetriever] = None):
        """
        Extend an agent instance with knowledge brain query capability.

        Adds a query_knowledge_brain() method to the agent.

        Args:
            agent: Agent instance to extend
            knowledge_retriever: Optional retriever (uses self.knowledge_retriever if not provided)
        """
        retriever = knowledge_retriever or self.knowledge_retriever

        def query_knowledge_brain(query: str,
                                 domain: Optional[str] = None,
                                 top_k: int = 5) -> str:
            """
            Query the knowledge brain for relevant information.

            Args:
                query: Search query
                domain: Optional domain filter
                top_k: Number of results

            Returns:
                Formatted context string with relevant knowledge
            """
            try:
                domains = [domain] if domain else None

                retrieval_result = retriever.search(
                    query=query,
                    top_k=top_k,
                    domains=domains
                )

                if not retrieval_result.results:
                    return f"No relevant knowledge found for: {query}"

                return retrieval_result.to_agent_context(max_results=top_k)

            except Exception as e:
                logger.error(f"Agent knowledge query failed: {e}")
                return f"Knowledge query failed: {e}"

        # Attach method to agent instance
        agent.query_knowledge_brain = query_knowledge_brain
        logger.debug(f"Extended agent {agent.agent_id} with knowledge brain capability")

    def extend_all_agents(self, agents: List[Any]):
        """
        Extend multiple agents with knowledge brain capability.

        Args:
            agents: List of agent instances
        """
        for agent in agents:
            self.extend_agent_with_knowledge_brain(agent)

        logger.info(f"Extended {len(agents)} agents with knowledge brain capability")


def integrate_with_felix_workflow(felix_system,
                                  knowledge_retriever: KnowledgeRetriever,
                                  auto_augment: bool = True) -> KnowledgeBrainIntegration:
    """
    Helper function to integrate knowledge brain with Felix system.

    Usage example:
    ```python
    from src.knowledge.retrieval import KnowledgeRetriever
    from src.knowledge.workflow_integration import integrate_with_felix_workflow

    # Initialize retriever
    retriever = KnowledgeRetriever(knowledge_store, embedding_provider)

    # Integrate with Felix
    integration = integrate_with_felix_workflow(felix_system, retriever)

    # In workflow, before spawning agents:
    augmented_context = integration.augment_workflow_context(
        task_description="Optimize this Python function",
        task_type="code_optimization"
    )

    # After workflow completes:
    integration.record_workflow_outcome(
        workflow_id=workflow_id,
        task_type="code_optimization",
        workflow_success=True,
        final_confidence=0.85
    )
    ```

    Args:
        felix_system: Felix system instance
        knowledge_retriever: KnowledgeRetriever instance
        auto_augment: Enable automatic augmentation

    Returns:
        KnowledgeBrainIntegration instance
    """
    integration = KnowledgeBrainIntegration(knowledge_retriever)

    logger.info("Knowledge brain integrated with Felix workflow system")
    logger.info(f"Auto-augmentation: {'enabled' if auto_augment else 'disabled'}")

    return integration


# Convenience function for adding knowledge brain to existing workflows
def add_knowledge_context_to_workflow_input(task_input: str,
                                           knowledge_retriever: KnowledgeRetriever,
                                           task_type: Optional[str] = None,
                                           max_concepts: int = 10) -> str:
    """
    Add knowledge context to workflow input.

    Simple wrapper that prepends relevant knowledge to task input.

    Args:
        task_input: Original task input
        knowledge_retriever: KnowledgeRetriever instance
        task_type: Optional task type
        max_concepts: Maximum concepts to include

    Returns:
        Augmented task input with knowledge context
    """
    try:
        knowledge_context = knowledge_retriever.build_augmented_context(
            task_description=task_input,
            task_type=task_type,
            max_concepts=max_concepts
        )

        if knowledge_context:
            return knowledge_context + "\n\n" + task_input
        else:
            return task_input

    except Exception as e:
        logger.error(f"Failed to add knowledge context: {e}")
        return task_input

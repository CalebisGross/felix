"""
Communication system for the Felix Framework.

This module implements both spoke-based and mesh communication architectures
for the helix system and comparison framework respectively.

Spoke Architecture:
- CentralPost: Hub for all agent communication
- Spoke: Individual communication channels
- O(N) scaling characteristics

Mesh Architecture:
- MeshCommunication: All-to-all topology for comparison
- MeshConnection: Pairwise agent connections  
- O(N²) scaling characteristics for Hypothesis H2 validation

The system supports:
- Reliable message delivery in both architectures
- Performance metrics collection and comparison
- Statistical analysis for research validation
- Scalable agent connections
"""

from .central_post import CentralPost, Message, MessageType
from .spoke import Spoke, SpokeConnection, SpokeManager
from .mesh import MeshCommunication, MeshConnection, MeshMessage

__all__ = [
    'CentralPost',
    'Message',
    'MessageType', 
    'Spoke',
    'SpokeConnection',
    'SpokeManager',
    'MeshCommunication',
    'MeshConnection',
    'MeshMessage'
]
"""
Spoke-based communication channels for the Felix Framework.

Implements the spoke connections between individual agents and the central post,
following the geometric model from thefelix.md where agents communicate along
radial spokes to the central coordination system.

Key Features:
- Bidirectional message passing (agent <-> central post)
- Reliable delivery with confirmation tracking
- Connection lifecycle management
- Message ordering and delivery guarantees
"""

import time
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from collections import deque

from .central_post import CentralPost, Message, MessageType


@dataclass
class DeliveryConfirmation:
    """Delivery confirmation for reliable messaging."""
    message_id: str
    delivery_time: float
    confirmed: bool = True


class SpokeConnection:
    """
    Base spoke connection interface.
    
    Defines the contract for spoke-based communication between
    agents and the central post.
    """
    
    def send_message(self, message: Message) -> str:
        """Send message through spoke connection."""
        raise NotImplementedError
    
    def receive_message(self, message: Message) -> str:
        """Receive message through spoke connection."""
        raise NotImplementedError
    
    def disconnect(self) -> None:
        """Disconnect the spoke."""
        raise NotImplementedError
    
    def reconnect(self) -> None:
        """Reconnect the spoke."""
        raise NotImplementedError


class Spoke(SpokeConnection):
    """
    Spoke communication channel between an agent and central post.
    
    Implements reliable, bidirectional communication following the
    geometric spoke model where agents connect radially to the central hub.
    """
    
    def __init__(self, agent, central_post: CentralPost):
        """
        Initialize spoke connection.
        
        Args:
            agent: Agent instance for this spoke
            central_post: Central post instance to connect to
        """
        self.agent = agent
        self.central_post = central_post
        self.agent_id = agent.agent_id
        
        # Connection state
        self._is_connected = False
        self._connection_id: Optional[str] = None
        
        # Message tracking
        self._messages_sent = 0
        self._messages_received = 0
        self._received_messages: deque = deque()
        self._delivery_confirmations: Dict[str, DeliveryConfirmation] = {}
        
        # Performance tracking
        self._send_times: Dict[str, float] = {}
        self._delivery_times: Dict[str, float] = {}
        
        # Establish initial connection
        self._connect()
    
    def _connect(self) -> None:
        """Establish connection to central post."""
        try:
            self._connection_id = self.central_post.register_agent(self.agent)
            self._is_connected = True
        except ValueError as e:
            self._is_connected = False
            raise RuntimeError(f"Failed to connect spoke for {self.agent_id}: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if spoke is currently connected."""
        return self._is_connected and self.central_post.is_agent_registered(self.agent_id)
    
    @property
    def message_count(self) -> int:
        """Get total number of messages sent through this spoke."""
        return self._messages_sent
    
    def send_message(self, message: Message) -> str:
        """
        Send message through spoke to central post.
        
        Args:
            message: Message to send
            
        Returns:
            Message ID for tracking
            
        Raises:
            RuntimeError: If spoke is not connected
        """
        if not self.is_connected:
            raise RuntimeError(f"Spoke for {self.agent_id} is not connected")
        
        # Validate message sender
        if message.sender_id != self.agent_id:
            raise ValueError("Message sender_id must match spoke agent_id")
        
        # Record send time for performance tracking
        send_time = time.time()
        self._send_times[message.message_id] = send_time
        
        # Send message to central post
        try:
            message_id = self.central_post.queue_message(message)
            self._messages_sent += 1
            
            # Create delivery confirmation (simulated immediate delivery)
            confirmation = DeliveryConfirmation(
                message_id=message_id,
                delivery_time=send_time
            )
            self._delivery_confirmations[message_id] = confirmation
            self._delivery_times[message_id] = send_time
            
            return message_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to send message through spoke: {e}")
    
    def send_message_reliable(self, message: Message) -> str:
        """
        Send message with guaranteed delivery confirmation.
        
        Args:
            message: Message to send
            
        Returns:
            Message ID with delivery guarantee
        """
        message_id = self.send_message(message)
        
        # For this implementation, delivery is immediate since we're using
        # direct queue communication. In a distributed system, this would
        # involve actual delivery confirmation protocols.
        
        return message_id
    
    def receive_message(self, message: Message) -> str:
        """
        Receive message from central post.
        
        Args:
            message: Message received from central post
            
        Returns:
            Receipt confirmation ID
        """
        if not self.is_connected:
            raise RuntimeError(f"Spoke for {self.agent_id} is not connected")
        
        # Validate message is addressed to this agent
        if hasattr(message, 'recipient_id'):
            if message.recipient_id != self.agent_id:
                raise ValueError("Message not addressed to this agent")
        
        # Store received message
        self._received_messages.append(message)
        self._messages_received += 1
        
        # Generate receipt confirmation
        receipt_id = str(uuid.uuid4())
        receipt_time = time.time()
        
        return receipt_id
    
    def get_received_messages(self) -> List[Message]:
        """
        Get all received messages.
        
        Returns:
            List of messages received through this spoke
        """
        return list(self._received_messages)
    
    def is_message_delivered(self, message_id: str) -> bool:
        """
        Check if a message has been delivered.
        
        Args:
            message_id: ID of message to check
            
        Returns:
            True if message was delivered, False otherwise
        """
        return message_id in self._delivery_confirmations
    
    def get_delivery_time(self, message_id: str) -> Optional[float]:
        """
        Get delivery time for a specific message.
        
        Args:
            message_id: ID of message to check
            
        Returns:
            Delivery timestamp, or None if not delivered
        """
        if message_id in self._delivery_times:
            return self._delivery_times[message_id]
        return None
    
    def disconnect(self) -> None:
        """Disconnect spoke from central post."""
        if self._is_connected:
            success = self.central_post.deregister_agent(self.agent_id)
            if success:
                self._is_connected = False
                self._connection_id = None
    
    def reconnect(self) -> None:
        """Reconnect spoke to central post."""
        if not self._is_connected:
            self._connect()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this spoke.
        
        Returns:
            Dictionary containing spoke performance data
        """
        total_messages = self._messages_sent + self._messages_received
        
        # Calculate average delivery time
        avg_delivery_time = 0.0
        if self._delivery_times:
            send_delivery_pairs = []
            for msg_id, delivery_time in self._delivery_times.items():
                if msg_id in self._send_times:
                    send_time = self._send_times[msg_id]
                    send_delivery_pairs.append(delivery_time - send_time)
            
            if send_delivery_pairs:
                avg_delivery_time = sum(send_delivery_pairs) / len(send_delivery_pairs)
        
        return {
            "agent_id": self.agent_id,
            "is_connected": self.is_connected,
            "messages_sent": self._messages_sent,
            "messages_received": self._messages_received,
            "total_messages": total_messages,
            "delivery_confirmations": len(self._delivery_confirmations),
            "average_delivery_time": avg_delivery_time,
            "connection_id": self._connection_id
        }
    
    def clear_received_messages(self) -> int:
        """
        Clear received message buffer.
        
        Returns:
            Number of messages cleared
        """
        count = len(self._received_messages)
        self._received_messages.clear()
        return count
    
    def __str__(self) -> str:
        """String representation for debugging."""
        connection_status = "connected" if self.is_connected else "disconnected"
        return (f"Spoke(agent={self.agent_id}, status={connection_status}, "
                f"sent={self._messages_sent}, received={self._messages_received})")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return str(self)


class SpokeManager:
    """
    Manager for multiple spoke connections.
    
    Provides utilities for managing communication across multiple agents
    and their spoke connections to the central post.
    """
    
    def __init__(self, central_post: CentralPost):
        """
        Initialize spoke manager.
        
        Args:
            central_post: Central post instance for spoke connections
        """
        self.central_post = central_post
        self._spokes: Dict[str, Spoke] = {}
    
    def create_spoke(self, agent) -> Spoke:
        """
        Create and register a spoke for an agent.
        
        Args:
            agent: Agent instance to create spoke for
            
        Returns:
            Created spoke instance
        """
        if agent.agent_id in self._spokes:
            raise ValueError(f"Spoke already exists for agent {agent.agent_id}")
        
        spoke = Spoke(agent=agent, central_post=self.central_post)
        self._spokes[agent.agent_id] = spoke
        
        return spoke
    
    def get_spoke(self, agent_id: str) -> Optional[Spoke]:
        """
        Get spoke for a specific agent.
        
        Args:
            agent_id: ID of agent to get spoke for
            
        Returns:
            Spoke instance, or None if not found
        """
        return self._spokes.get(agent_id)
    
    def remove_spoke(self, agent_id: str) -> bool:
        """
        Remove and disconnect spoke for an agent.
        
        Args:
            agent_id: ID of agent to remove spoke for
            
        Returns:
            True if spoke was removed, False if not found
        """
        if agent_id not in self._spokes:
            return False
        
        spoke = self._spokes[agent_id]
        spoke.disconnect()
        del self._spokes[agent_id]
        
        return True
    
    def get_all_spokes(self) -> List[Spoke]:
        """
        Get all managed spokes.
        
        Returns:
            List of all spoke instances
        """
        return list(self._spokes.values())
    
    def broadcast_message(self, message: Message, exclude_agents: List[str] = None) -> List[str]:
        """
        Broadcast message to all connected spokes.
        
        Args:
            message: Message to broadcast
            exclude_agents: List of agent IDs to exclude from broadcast
            
        Returns:
            List of message IDs for tracking
        """
        exclude_agents = exclude_agents or []
        message_ids = []
        
        for agent_id, spoke in self._spokes.items():
            if agent_id not in exclude_agents and spoke.is_connected:
                try:
                    # Create copy of message with appropriate recipient
                    broadcast_msg = Message(
                        sender_id=message.sender_id,
                        message_type=message.message_type,
                        content=message.content.copy(),
                        timestamp=message.timestamp
                    )
                    
                    # Add recipient information
                    broadcast_msg.content['recipient_id'] = agent_id
                    
                    receipt_id = spoke.receive_message(broadcast_msg)
                    message_ids.append(receipt_id)
                    
                except Exception as e:
                    # Log error but continue with other spokes
                    print(f"Failed to broadcast to {agent_id}: {e}")
        
        return message_ids
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """
        Get summary of all spoke connections.
        
        Returns:
            Dictionary containing connection statistics
        """
        connected_count = sum(1 for spoke in self._spokes.values() if spoke.is_connected)
        total_messages_sent = sum(spoke.message_count for spoke in self._spokes.values())
        
        return {
            "total_spokes": len(self._spokes),
            "connected_spokes": connected_count,
            "disconnected_spokes": len(self._spokes) - connected_count,
            "total_messages_sent": total_messages_sent,
            "agent_ids": list(self._spokes.keys())
        }
    
    def register_agent(self, agent) -> str:
        """
        Register an agent and create a spoke connection.
        
        Args:
            agent: Agent to register
            
        Returns:
            Connection ID for the spoke
        """
        spoke = self.create_spoke(agent)
        return spoke._connection_id
    
    def process_all_messages(self) -> int:
        """
        Process and distribute messages through spoke system.
        
        Returns:
            Total number of messages processed
        """
        total_processed = 0
        
        # Process messages from central post and distribute to agents
        while self.central_post.has_pending_messages():
            message = self.central_post.process_next_message()
            if message:
                # Distribute to all active agents except sender
                for agent_id, spoke in self._spokes.items():
                    if (agent_id != message.sender_id and 
                        spoke.is_connected and 
                        hasattr(spoke.agent, 'receive_shared_context')):
                        # Send message content to agent via their receive method
                        spoke.agent.receive_shared_context(message.content)
                total_processed += 1
        
        return total_processed
    
    def send_message(self, agent_id: str, message) -> str:
        """
        Send message from specific agent through their spoke.
        
        Args:
            agent_id: ID of the agent sending the message
            message: Message to send
            
        Returns:
            Message ID for tracking
            
        Raises:
            ValueError: If agent doesn't have a spoke
        """
        if agent_id not in self._spokes:
            raise ValueError(f"No spoke found for agent {agent_id}")
        
        spoke = self._spokes[agent_id]
        return spoke.send_message(message)
    
    def shutdown_all(self) -> None:
        """Disconnect and remove all spokes."""
        for spoke in self._spokes.values():
            spoke.disconnect()
        self._spokes.clear()
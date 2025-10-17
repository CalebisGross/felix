"""
Helix geometry calculations for the Felix Framework.

This module implements the core mathematical model for the helical agent path,
translating the 3D geometric model from thefelix.md into computational form.

Mathematical Foundation:
- Parametric helix with exponential radius tapering
- Position vector r(t) = (R(t)cos(θ(t)), R(t)sin(θ(t)), Ht)
- Parameter t ∈ [0,1] where t=0 is bottom, t=1 is top
- Tapering function R(t) = R_bottom * (R_top/R_bottom)^t
- Angular function θ(t) = 2πnt where n is number of turns

For complete mathematical specification, see:
- docs/mathematical_model.md: Formal parametric equations and geometric properties
- docs/hypothesis_mathematics.md: Statistical formulations for research hypotheses
- thefelix.md: Original OpenSCAD geometric prototype
- validate_openscad.py: Numerical validation against OpenSCAD (<1e-12 precision)

Implementation validates against OpenSCAD model with mathematical precision.
"""

import math
from typing import Tuple


class HelixGeometry:
    """
    Core helix mathematical model for agent positioning.
    
    Implements the same parametric equations as the OpenSCAD prototype,
    allowing validation against the geometric visualization.
    """
    
    def __init__(self, top_radius: float, bottom_radius: float, height: float, turns: int):
        """
        Initialize helix with geometric parameters.
        
        Args:
            top_radius: Radius at the top of the helix (t=0)
            bottom_radius: Radius at the bottom of the helix (t=1)  
            height: Total vertical height of the helix
            turns: Number of complete rotations
            
        Raises:
            ValueError: If parameters are invalid
        """
        self._validate_parameters(top_radius, bottom_radius, height, turns)
        
        self.top_radius = top_radius
        self.bottom_radius = bottom_radius
        self.height = height
        self.turns = turns
    
    def _validate_parameters(self, top_radius: float, bottom_radius: float, 
                           height: float, turns: int) -> None:
        """Validate helix parameters for mathematical consistency."""
        if top_radius <= bottom_radius:
            raise ValueError("top_radius must be greater than bottom_radius")
        
        if height <= 0:
            raise ValueError("height must be positive")
            
        if turns <= 0:
            raise ValueError("turns must be positive")
    
    def get_position(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate 3D position along helix path.
        
        Implements the parametric helix equation:
        r(t) = (R(t)cos(θ(t)), R(t)sin(θ(t)), Ht)
        
        Where:
        - R(t) = R_bottom * (R_top/R_bottom)^t (exponential tapering)
        - θ(t) = 2πnt (angular progression)
        - z(t) = Ht (linear height progression)
        
        Mathematical reference: docs/mathematical_model.md, Section 1.2
        
        Args:
            t: Parameter value between 0 (bottom) and 1 (top)
            
        Returns:
            Tuple of (x, y, z) coordinates
            
        Raises:
            ValueError: If t is outside [0,1] range
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")
        
        # Calculate height (linear interpolation: t=0 is bottom, t=1 is top)
        # Matches OpenSCAD model where step=0 is bottom, step=max is top
        z = self.height * t
        
        # Calculate radius at this height (exponential tapering)
        radius = self.get_radius(z)
        
        # Calculate angle (linear progression through turns)
        # Total rotation: turns * 360 degrees = turns * 2π radians
        angle_radians = t * self.turns * 2.0 * math.pi
        
        # Calculate Cartesian coordinates
        x = radius * math.cos(angle_radians)
        y = radius * math.sin(angle_radians)
        
        return (x, y, z)
    
    def get_radius(self, z: float) -> float:
        """
        Calculate radius at given height using exponential tapering.
        
        Implements the tapering function:
        R(z) = R_bottom * (R_top/R_bottom)^(z/height)
        
        This creates exponential tapering that naturally focuses agent density
        toward the narrow end, supporting Hypothesis H3 (attention focusing).
        
        Mathematical reference: docs/mathematical_model.md, Section 2
        Hypothesis reference: docs/hypothesis_mathematics.md, Section H3.2
        
        Args:
            z: Height value (0 = bottom, height = top)
            
        Returns:
            Radius at the specified height
        """
        # Ensure z is within valid range
        z = max(0.0, min(z, self.height))
        
        # Exponential tapering formula from OpenSCAD
        radius_ratio = self.top_radius / self.bottom_radius
        height_fraction = z / self.height
        radius = self.bottom_radius * pow(radius_ratio, height_fraction)
        
        return radius
    
    def get_angle_at_t(self, t: float) -> float:
        """
        Calculate rotation angle (in radians) at parameter t.
        
        Args:
            t: Parameter value between 0 and 1
            
        Returns:
            Angle in radians
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")
            
        return t * self.turns * 2.0 * math.pi
    
    def get_tangent_vector(self, t: float) -> Tuple[float, float, float]:
        """
        Calculate tangent vector to helix at parameter t.
        
        Useful for agent orientation and movement direction.
        
        Args:
            t: Parameter value between 0 and 1
            
        Returns:
            Normalized tangent vector (dx/dt, dy/dt, dz/dt)
        """
        if not (0.0 <= t <= 1.0):
            raise ValueError("t must be between 0 and 1")
        
        # Small epsilon for numerical differentiation
        eps = 1e-8
        t1 = max(0.0, t - eps)
        t2 = min(1.0, t + eps)
        
        x1, y1, z1 = self.get_position(t1)
        x2, y2, z2 = self.get_position(t2)
        
        # Calculate direction vector
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy + dz*dz)
        if length > 0:
            dx /= length
            dy /= length
            dz /= length
        
        return (dx, dy, dz)
    
    def approximate_arc_length(self, t_start: float = 0.0, t_end: float = 1.0, 
                              segments: int = 1000) -> float:
        """
        Approximate arc length of helix segment using linear interpolation.
        
        Args:
            t_start: Starting parameter value
            t_end: Ending parameter value
            segments: Number of segments for approximation
            
        Returns:
            Approximate arc length
        """
        if not (0.0 <= t_start <= t_end <= 1.0):
            raise ValueError("Invalid t_start or t_end values")
        
        if segments < 1:
            raise ValueError("segments must be positive")
        
        total_length = 0.0
        dt = (t_end - t_start) / segments
        
        prev_x, prev_y, prev_z = self.get_position(t_start)
        
        for i in range(1, segments + 1):
            t = t_start + i * dt
            x, y, z = self.get_position(t)
            
            # Calculate distance from previous point
            distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)
            total_length += distance
            
            prev_x, prev_y, prev_z = x, y, z
        
        return total_length
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"HelixGeometry(top_radius={self.top_radius}, "
                f"bottom_radius={self.bottom_radius}, "
                f"height={self.height}, turns={self.turns})")
# Core Module

## Purpose
Fundamental mathematical models providing the geometric foundation for Felix's helical agent progression system.

## Key Files

### [helix_geometry.py](helix_geometry.py)
Parametric helix modeling for agent positioning and behavior adaptation.
- **`HelixGeometry`**: Mathematical model of 3D spiral progression

## Key Concepts

### Helical Agent Progression

Agents move along a 3D helix (spiral) from exploration (top) to synthesis (bottom), with their behavior adapting based on position:

**Helix Parameters** (default):
- **Top radius**: 3.0 - Wide exploration phase
- **Bottom radius**: 0.5 - Narrow synthesis phase
- **Height**: 8.0 - Total progression depth
- **Turns**: 2 - Spiral complexity (2 complete rotations)

### Mathematical Model

Parametric helix equations with exponential radius tapering:

```
x(t) = r(t) × cos(2πn × t)
y(t) = r(t) × sin(2πn × t)
z(t) = h × t

where:
  t = normalized time (0.0 to 1.0)
  n = number of turns (default: 2)
  h = height (default: 8.0)
  r(t) = top_radius × e^(-α×t) + bottom_radius × (1 - e^(-α×t))
  α = taper rate
```

**Key Method**: `get_position(normalized_time: float) -> tuple[float, float, float]`
- Input: Normalized time 0.0 (start) to 1.0 (end)
- Output: (x, y, z) coordinates on helix

### Position-Based Adaptation

Agent behavior changes based on helix position:

#### 1. Temperature Gradient
- **Top (t=0.0)**: temperature = 1.0 (creative, exploratory)
- **Bottom (t=1.0)**: temperature = 0.2 (focused, deterministic)
- **Interpolation**: Linear gradient between positions

#### 2. Token Budget
- **Top**: Larger budget for broad exploration
- **Middle**: Moderate budget for focused analysis
- **Bottom**: Smaller budget for concise synthesis

#### 3. Agent Role Specialization
- **Early (t=0.0-0.25)**: Research agents spawn
- **Mid (t=0.2-0.6)**: Analysis agents spawn
- **Late (t=0.4-0.7)**: Critic agents spawn

#### 4. Spatial Radius
- **Wide (top)**: Broad exploration, diverse approaches
- **Narrow (bottom)**: Converged consensus, unified output

### Hypothesis Validation

**H1: Helical progression enhances agent adaptation**
- Target: 20% improvement in workload distribution
- Mechanism: Progressive specialization from exploration to synthesis

### Configuration

```yaml
helix:
  top_radius: 3.0      # Exploration breadth
  bottom_radius: 0.5   # Synthesis focus
  height: 8.0          # Progression depth
  turns: 2             # Spiral complexity
```

### Usage Example

```python
from src.core.helix_geometry import HelixGeometry

# Initialize helix with custom parameters
helix = HelixGeometry(
    top_radius=3.0,
    bottom_radius=0.5,
    height=8.0,
    turns=2
)

# Get agent position at 25% progress
x, y, z = helix.get_position(normalized_time=0.25)

# Calculate temperature for this position
temperature = 1.0 - (0.8 * normalized_time)  # 1.0 → 0.2
```

## Geometric Visualization

```
     Top (t=0.0, r=3.0)
         .-''-.
       .'       '.
      /           \      ← Wide exploration
     |             |
     |             |
      \           /       ← Tapering
       '.       .'
         '-...-'          ← Narrow synthesis
   Bottom (t=1.0, r=0.5)

Height: 8.0 units
Turns: 2 complete spirals
```

## Related Modules
- [agents/](../agents/) - LLMAgent uses helix positioning for behavior adaptation
- [communication/](../communication/) - AgentFactory positions agents on helix
- [llm/](../llm/) - Temperature gradient derived from helix position
- [workflows/](../workflows/) - Workflow progression follows helix model

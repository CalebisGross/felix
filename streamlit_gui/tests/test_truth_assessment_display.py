"""Tests for TruthAssessmentDisplay component."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from streamlit_gui.components.truth_assessment_display import TruthAssessmentDisplay


def test_truth_keyword_detection():
    """Test detection of truth assessment workflows."""

    # Workflow with truth keywords
    truth_workflow = {
        'task_input': 'Validate and verify the accuracy of this information',
        'confidence': 0.9
    }
    assert TruthAssessmentDisplay._is_truth_assessment_workflow(truth_workflow)

    # Workflow without truth keywords
    normal_workflow = {
        'task_input': 'Analyze the market trends',
        'confidence': 0.8
    }
    assert not TruthAssessmentDisplay._is_truth_assessment_workflow(normal_workflow)

    # Edge cases
    truth_workflow2 = {
        'task_input': 'Check accuracy of these facts',
        'confidence': 0.7
    }
    assert TruthAssessmentDisplay._is_truth_assessment_workflow(truth_workflow2)


def test_validation_status():
    """Test validation status determination."""

    # High confidence - Validated
    high_conf = {'confidence': 0.90}
    status, badge, color = TruthAssessmentDisplay._determine_validation_status(high_conf)
    assert status == 'validated'
    assert '✓' in badge
    assert color == 'success'

    # Medium confidence - Needs Review
    med_conf = {'confidence': 0.75}
    status, badge, color = TruthAssessmentDisplay._determine_validation_status(med_conf)
    assert status == 'needs_review'
    assert '⚠' in badge
    assert color == 'warning'

    # Low confidence - Failed
    low_conf = {'confidence': 0.65}
    status, badge, color = TruthAssessmentDisplay._determine_validation_status(low_conf)
    assert status == 'failed'
    assert '✗' in badge
    assert color == 'error'

    # Edge cases
    edge_high = {'confidence': 0.85}
    status, _, _ = TruthAssessmentDisplay._determine_validation_status(edge_high)
    assert status == 'validated'

    edge_med = {'confidence': 0.70}
    status, _, _ = TruthAssessmentDisplay._determine_validation_status(edge_med)
    assert status == 'needs_review'


def test_source_extraction():
    """Test extraction of sources from synthesis."""

    synthesis1 = "According to Wikipedia and from NASA data, based on scientific research..."
    sources = TruthAssessmentDisplay._extract_sources_from_synthesis(synthesis1)
    assert len(sources) > 0
    assert any('wikipedia' in s.lower() for s in sources)

    synthesis2 = "The information is correct."
    sources = TruthAssessmentDisplay._extract_sources_from_synthesis(synthesis2)
    assert isinstance(sources, list)

    # Empty synthesis
    sources = TruthAssessmentDisplay._extract_sources_from_synthesis("")
    assert sources == []


def test_reasoning_extraction():
    """Test extraction of reasoning snippets."""

    synthesis_with_assessment = """
    Assessment: This information has been validated by multiple authoritative sources
    and shows high confidence in accuracy.
    """
    reasoning = TruthAssessmentDisplay._extract_reasoning_snippet(synthesis_with_assessment)
    assert reasoning is not None
    assert 'assessment' in reasoning.lower()

    # Short synthesis
    short_synthesis = "Valid information."
    reasoning = TruthAssessmentDisplay._extract_reasoning_snippet(short_synthesis)
    assert reasoning is not None

    # Long synthesis without specific patterns
    long_synthesis = "This is a very long synthesis " * 20
    reasoning = TruthAssessmentDisplay._extract_reasoning_snippet(long_synthesis)
    assert reasoning is not None
    assert len(reasoning) <= 155  # 150 chars + '...'


def test_comprehensive_workflow():
    """Test complete workflow processing."""

    # Create a realistic truth assessment workflow
    workflow = {
        'workflow_id': 42,
        'task_input': 'Validate the current time and date information for accuracy',
        'confidence': 0.88,
        'agents_count': 5,
        'status': 'completed',
        'final_synthesis': '''
        Assessment: The information has been validated through multiple sources.
        According to time.is and from worldtimeserver.com, the current time is accurate.
        Confidence based on consensus from authoritative time sources.
        '''
    }

    # Test keyword detection
    assert TruthAssessmentDisplay._is_truth_assessment_workflow(workflow)

    # Test validation status
    status, badge, color = TruthAssessmentDisplay._determine_validation_status(workflow)
    assert status == 'validated'
    assert color == 'success'

    # Test source extraction
    sources = TruthAssessmentDisplay._extract_sources_from_synthesis(workflow['final_synthesis'])
    assert len(sources) > 0

    # Test reasoning extraction
    reasoning = TruthAssessmentDisplay._extract_reasoning_snippet(workflow['final_synthesis'])
    assert reasoning is not None
    assert 'assessment' in reasoning.lower()


if __name__ == '__main__':
    print("Running TruthAssessmentDisplay tests...")

    test_truth_keyword_detection()
    print("[PASS] Truth keyword detection tests passed")

    test_validation_status()
    print("[PASS] Validation status tests passed")

    test_source_extraction()
    print("[PASS] Source extraction tests passed")

    test_reasoning_extraction()
    print("[PASS] Reasoning extraction tests passed")

    test_comprehensive_workflow()
    print("[PASS] Comprehensive workflow tests passed")

    print("\nAll tests passed successfully!")

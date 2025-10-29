"""Test script for Phase 3 components."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_truth_assessment_component():
    """Test truth assessment display component."""
    print("Testing truth assessment display component...")

    from streamlit_gui.components.truth_assessment_display import TruthAssessmentDisplay

    # Test workflow detection
    test_workflow = {
        'task_input': 'Validate the accuracy of this information',
        'confidence': 0.88,
        'final_synthesis': 'According to Time.is and WorldTimeServer.com, the information is correct.'
    }

    is_truth = TruthAssessmentDisplay._is_truth_assessment_workflow(test_workflow)
    assert is_truth, "Should detect truth assessment workflow"

    status, badge, color = TruthAssessmentDisplay._determine_validation_status(test_workflow)
    assert status == 'validated', f"Should be validated, got {status}"
    assert badge == '✓ Validated', f"Badge should be validated, got {badge}"

    print("[PASS] Truth assessment component tests passed")

def test_configuration_page():
    """Test configuration page syntax."""
    print("Testing configuration page...")

    # Just check if file compiles
    import py_compile
    py_compile.compile('streamlit_gui/pages/2_Configuration.py', doraise=True)

    print("[PASS] Configuration page syntax valid")

def test_dashboard_page():
    """Test dashboard page syntax."""
    print("Testing dashboard page...")

    # Just check if file compiles
    import py_compile
    py_compile.compile('streamlit_gui/pages/1_Dashboard.py', doraise=True)

    print("[PASS] Dashboard page syntax valid")

def test_workflow_history_integration():
    """Test workflow history viewer integration."""
    print("Testing workflow history viewer integration...")

    from streamlit_gui.components.workflow_history_viewer import WorkflowHistoryViewer

    # Check that the class exists and has the render method
    assert hasattr(WorkflowHistoryViewer, 'render'), "WorkflowHistoryViewer should have render method"

    print("[PASS] Workflow history viewer integration valid")

if __name__ == '__main__':
    print("=" * 60)
    print("Phase 3 Component Tests")
    print("=" * 60)

    try:
        test_truth_assessment_component()
        test_configuration_page()
        test_dashboard_page()
        test_workflow_history_integration()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
        print("\nPhase 3 components are ready for use!")

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Truth Assessment Display component for Streamlit GUI."""

import streamlit as st
from typing import Dict, Any, Optional
import re


class TruthAssessmentDisplay:
    """Display truth assessment validation badges and details for workflows."""

    # Keywords that indicate truth assessment was used
    TRUTH_KEYWORDS = [
        'truth', 'validate', 'verify', 'assess', 'check accuracy',
        'fact-check', 'truthful', 'validation', 'verification'
    ]

    @staticmethod
    def _is_truth_assessment_workflow(workflow: Dict[str, Any]) -> bool:
        """Check if workflow used truth assessment based on task keywords."""
        task_input = workflow.get('task_input', '').lower()
        return any(keyword in task_input for keyword in TruthAssessmentDisplay.TRUTH_KEYWORDS)

    @staticmethod
    def _determine_validation_status(workflow: Dict[str, Any]) -> tuple[str, str, str]:
        """
        Determine validation status based on confidence threshold.

        Returns:
            Tuple of (status, badge_text, badge_color)
        """
        confidence = workflow.get('confidence', 0.0)

        # High confidence (>=0.85) = Validated
        if confidence >= 0.85:
            return ('validated', '✓ Validated', 'success')
        # Medium confidence (0.70-0.84) = Needs Review
        elif confidence >= 0.70:
            return ('needs_review', '⚠ Needs Review', 'warning')
        # Low confidence (<0.70) = Failed Validation
        else:
            return ('failed', '✗ Failed Validation', 'error')

    @staticmethod
    def _extract_sources_from_synthesis(synthesis: str) -> list[str]:
        """Extract source information from synthesis output."""
        sources = []

        # Look for common source patterns in synthesis
        # Pattern: "According to [source]"
        according_to = re.findall(r'according to ([^,.\n]+)', synthesis.lower())
        sources.extend(according_to)

        # Pattern: "From [source]"
        from_sources = re.findall(r'from ([^,.\n]+)', synthesis.lower())
        sources.extend(from_sources)

        # Pattern: "Based on [source]"
        based_on = re.findall(r'based on ([^,.\n]+)', synthesis.lower())
        sources.extend(based_on)

        # Clean and deduplicate
        sources = [s.strip().title() for s in sources if s.strip()]
        return list(set(sources))[:5]  # Return up to 5 unique sources

    @staticmethod
    def _extract_reasoning_snippet(synthesis: str) -> Optional[str]:
        """Extract assessment reasoning from synthesis output."""
        if not synthesis or not synthesis.strip():
            return None

        # Look for reasoning patterns
        reasoning_patterns = [
            r'(assessment: .{50,200})',
            r'(validation shows .{50,200})',
            r'(confidence based on .{50,200})',
            r'(verified through .{50,200})',
        ]

        for pattern in reasoning_patterns:
            match = re.search(pattern, synthesis.lower(), re.IGNORECASE)
            if match:
                return match.group(1).capitalize() + '...'

        # If no specific pattern, return first 150 chars or entire text if shorter
        if len(synthesis) > 150:
            return synthesis[:150] + '...'
        elif len(synthesis) > 0:
            return synthesis.strip()
        return None

    @staticmethod
    def render_assessment_badge(workflow: Dict[str, Any]) -> None:
        """Render compact assessment badge below status header."""
        # Only show if this appears to be a truth assessment workflow
        if not TruthAssessmentDisplay._is_truth_assessment_workflow(workflow):
            return

        status, badge_text, badge_color = TruthAssessmentDisplay._determine_validation_status(workflow)

        # Display badge with confidence score
        confidence_pct = workflow.get('confidence', 0.0) * 100

        if badge_color == 'success':
            st.success(f"🔍 **Truth Assessment**: {badge_text} (Confidence: {confidence_pct:.1f}%)")
        elif badge_color == 'warning':
            st.warning(f"🔍 **Truth Assessment**: {badge_text} (Confidence: {confidence_pct:.1f}%)")
        else:
            st.error(f"🔍 **Truth Assessment**: {badge_text} (Confidence: {confidence_pct:.1f}%)")

    @staticmethod
    def render_assessment_details(workflow: Dict[str, Any]) -> None:
        """Render detailed assessment information in expandable section."""
        # Only show if this appears to be a truth assessment workflow
        if not TruthAssessmentDisplay._is_truth_assessment_workflow(workflow):
            return

        with st.expander("📊 View Truth Assessment Details"):
            st.markdown("#### Validation Analysis")

            # Status and confidence
            status, badge_text, badge_color = TruthAssessmentDisplay._determine_validation_status(workflow)
            confidence = workflow.get('confidence', 0.0)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Validation Status", badge_text)

            with col2:
                st.metric("Confidence Score", f"{confidence*100:.1f}%")

            with col3:
                agents = workflow.get('agents_count', 0)
                st.metric("Validation Agents", agents)

            # Extract verification sources
            synthesis = workflow.get('final_synthesis', '')
            sources = TruthAssessmentDisplay._extract_sources_from_synthesis(synthesis)

            if sources:
                st.markdown("#### Verification Sources")
                for i, source in enumerate(sources, 1):
                    st.markdown(f"{i}. {source}")

            # Assessment reasoning
            reasoning = TruthAssessmentDisplay._extract_reasoning_snippet(synthesis)
            if reasoning:
                st.markdown("#### Assessment Reasoning")
                st.info(reasoning)

            # Interpretation guide
            st.markdown("#### Confidence Threshold Guide")
            st.markdown("""
            - **≥85%**: High confidence - Information validated by multiple sources
            - **70-84%**: Medium confidence - Partial validation, review recommended
            - **<70%**: Low confidence - Insufficient validation, requires verification
            """)

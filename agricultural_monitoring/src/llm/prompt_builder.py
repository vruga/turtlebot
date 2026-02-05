#!/usr/bin/env python3
"""
Prompt Builder for Claude LLM Integration

Builds context-aware prompts for farmer recommendations based on:
- Disease detection results
- Detection history patterns
- Time of day and environmental factors

Author: Agricultural Robotics Team
License: MIT
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for Claude API based on detection context.

    Prompt selection:
    - First detection of day: More detailed explanation
    - Repeated same disease: Pattern recognition prompt
    - Multiple different diseases: Environmental stress prompt
    - High confidence (>95%): Confident treatment advice
    - Low confidence (80-90%): Suggest manual verification

    Attributes:
        templates: Dict of prompt templates from config
        system_prompt: Base system prompt for all requests
    """

    def __init__(self, config_path: Optional[Path] = None) -> None:
        """
        Initialize the prompt builder.

        Args:
            config_path: Path to llm_config.yaml
        """
        self.templates = {}
        self.system_prompt = ""
        self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path]) -> None:
        """Load prompt templates from config."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'llm_config.yaml'

        # Default system prompt
        self.system_prompt = (
            "You are an agricultural expert assistant helping farmers identify "
            "and treat plant diseases. Provide practical, actionable advice in "
            "simple language. Keep responses concise and farmer-friendly."
        )

        # Default templates
        self.templates = {
            'standard': self._default_standard_template(),
            'first_detection': self._default_first_detection_template(),
            'repeated_disease': self._default_repeated_disease_template(),
            'multiple_diseases': self._default_multiple_diseases_template(),
            'high_confidence': self._default_high_confidence_template(),
            'low_confidence': self._default_low_confidence_template()
        }

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                return

            prompts = config.get('prompts', {})

            if 'system_prompt' in prompts:
                self.system_prompt = prompts['system_prompt']

            # Load templates
            template_keys = [
                ('standard_template', 'standard'),
                ('first_detection_template', 'first_detection'),
                ('repeated_disease_template', 'repeated_disease'),
                ('multiple_diseases_template', 'multiple_diseases'),
                ('high_confidence_template', 'high_confidence'),
                ('low_confidence_template', 'low_confidence')
            ]

            for config_key, template_key in template_keys:
                if config_key in prompts:
                    self.templates[template_key] = prompts[config_key]

        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _default_standard_template(self) -> str:
        return """Disease detected: {disease_name}
Confidence: {confidence_percent}%
Severity: {severity}

Our automated system has applied a {spray_duration}ms spray treatment.

Please provide concise farmer-friendly advice:
1. Confirm disease identification
2. Evaluate if spray treatment was appropriate
3. Additional treatment if needed
4. Preventive measures
5. Expected recovery timeline

Keep response under 200 words, use simple language."""

    def _default_first_detection_template(self) -> str:
        return """Good morning! First disease detection of the day.

Disease detected: {disease_name}
Confidence: {confidence_percent}%
Time: {time_of_day}

We applied a {spray_duration}ms spray treatment.

Please provide a morning briefing:
1. Explain this disease and typical progression
2. Why it might have appeared today
3. What to watch for during the day
4. Treatment expectations
5. Warning signs that need attention

Keep response under 250 words."""

    def _default_repeated_disease_template(self) -> str:
        return """PATTERN ALERT: Same disease detected multiple times

Disease: {disease_name}
Detection count today: {repeat_count}
Time span: {time_span}
Confidence range: {confidence_range}%

This repeated detection suggests a systemic issue.

Please analyze:
1. Why this disease keeps appearing
2. Are we treating symptoms, not root cause?
3. Environmental factors to address
4. Should we adjust our approach?
5. When to consider professional help

Be direct about whether current treatment is working."""

    def _default_multiple_diseases_template(self) -> str:
        return """WARNING: Multiple different diseases detected

Diseases found today: {disease_list}
Total detections: {total_count}

Multiple disease types suggest environmental stress.

Please advise:
1. What conditions cause multiple diseases?
2. Priority order for treatment
3. Are these diseases related?
4. Field-wide vs spot treatment
5. Should we pause and assess?"""

    def _default_high_confidence_template(self) -> str:
        return """HIGH CONFIDENCE DETECTION ({confidence_percent}%)

Disease: {disease_name}
Severity: {severity}

With this confidence level, provide definitive advice:
1. Confirm this is {disease_name}
2. Optimal treatment protocol
3. Expected results timeline
4. Follow-up actions at 24/48/72 hours

Be confident and direct in your recommendations."""

    def _default_low_confidence_template(self) -> str:
        return """UNCERTAIN DETECTION - Manual verification recommended

Possible disease: {disease_name}
Confidence: {confidence_percent}% (below typical threshold)

Because confidence is lower than usual:
1. What other diseases look similar?
2. Key visual differences to check manually
3. Should farmer inspect this plant closely?
4. Is precautionary treatment wise?
5. What additional information would help?

Emphasize the need for human verification."""

    def build_prompt(
        self,
        detection: Dict[str, Any],
        history: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> str:
        """
        Build appropriate prompt based on detection and context.

        Args:
            detection: Current detection result
            history: Recent detection history
            context: Additional context (time, counts, etc.)

        Returns:
            Formatted prompt string
        """
        disease_name = detection.get('disease_name', 'unknown')
        confidence = detection.get('confidence', 0)
        severity = detection.get('severity', 'unknown')
        spray_duration = detection.get('spray_duration', 0)

        # Determine which template to use
        template_key = self._select_template(detection, history, context)
        template = self.templates.get(template_key, self.templates['standard'])

        # Build format values
        format_values = {
            'disease_name': disease_name,
            'confidence': confidence,
            'confidence_percent': round(confidence * 100, 1),
            'severity': severity,
            'spray_duration': spray_duration,
            'time_of_day': context.get('time_of_day', datetime.now().strftime('%H:%M'))
        }

        # Add context-specific values
        if template_key == 'repeated_disease':
            format_values.update(self._get_repeated_context(disease_name, history))
        elif template_key == 'multiple_diseases':
            format_values.update(self._get_multiple_context(history))
        elif template_key == 'first_detection':
            format_values.update(self._get_first_detection_context())

        # Format template
        try:
            return template.format(**format_values)
        except KeyError as e:
            logger.warning(f"Missing template value: {e}")
            return self.templates['standard'].format(**format_values)

    def _select_template(
        self,
        detection: Dict,
        history: List[Dict],
        context: Dict
    ) -> str:
        """Select the most appropriate template based on context."""
        disease_name = detection.get('disease_name', 'unknown')
        confidence = detection.get('confidence', 0)

        # Check for first detection of day
        if context.get('is_first_today', False):
            return 'first_detection'

        # Check for low confidence
        if 0.80 <= confidence < 0.90:
            return 'low_confidence'

        # Check for high confidence
        if confidence >= 0.95:
            return 'high_confidence'

        # Check for repeated same disease
        recent_same = sum(
            1 for h in history[-10:]
            if h.get('disease_name') == disease_name
        )
        if recent_same >= 3:
            return 'repeated_disease'

        # Check for multiple different diseases
        recent_diseases = set(h.get('disease_name') for h in history[-10:])
        if len(recent_diseases) >= 3:
            return 'multiple_diseases'

        return 'standard'

    def _get_repeated_context(
        self,
        disease_name: str,
        history: List[Dict]
    ) -> Dict[str, Any]:
        """Get context values for repeated disease template."""
        same_disease = [
            h for h in history
            if h.get('disease_name') == disease_name
        ]

        if not same_disease:
            return {
                'repeat_count': 1,
                'time_span': 'N/A',
                'confidence_range': 'N/A'
            }

        confidences = [h.get('confidence', 0) for h in same_disease]

        # Calculate time span
        if len(same_disease) > 1:
            first = datetime.fromisoformat(same_disease[0].get('timestamp', ''))
            last = datetime.fromisoformat(same_disease[-1].get('timestamp', ''))
            time_span = str(last - first).split('.')[0]
        else:
            time_span = 'just now'

        return {
            'repeat_count': len(same_disease),
            'time_span': time_span,
            'confidence_range': f"{min(confidences)*100:.0f}-{max(confidences)*100:.0f}"
        }

    def _get_multiple_context(self, history: List[Dict]) -> Dict[str, Any]:
        """Get context values for multiple diseases template."""
        diseases = set()
        for h in history[-10:]:
            name = h.get('disease_name')
            if name and name != 'healthy':
                diseases.add(name)

        return {
            'disease_list': ', '.join(sorted(diseases)),
            'total_count': len(history)
        }

    def _get_first_detection_context(self) -> Dict[str, Any]:
        """Get context values for first detection template."""
        return {
            'yesterday_summary': 'No data available'
        }

    def get_system_prompt(self) -> str:
        """Get the system prompt for API calls."""
        return self.system_prompt

    def get_available_templates(self) -> List[str]:
        """Get list of available template keys."""
        return list(self.templates.keys())


if __name__ == '__main__':
    # Quick test
    builder = PromptBuilder()

    print("Available templates:", builder.get_available_templates())
    print("\nSystem prompt:", builder.get_system_prompt()[:100], "...")

    # Test prompt building
    detection = {
        'disease_name': 'early_blight',
        'confidence': 0.87,
        'severity': 'mild',
        'spray_duration': 2000,
        'timestamp': datetime.now().isoformat()
    }

    context = {
        'is_first_today': False,
        'time_of_day': '10:30'
    }

    prompt = builder.build_prompt(detection, [], context)
    print("\nGenerated prompt:")
    print(prompt)

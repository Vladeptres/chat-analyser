# Work Context Analysis

## Purpose
This context is designed for analyzing work-related conversations, including team communications, project discussions, meeting transcripts, and professional interactions.

## Analysis Focus Areas

### Communication Patterns
- **Meeting Efficiency**: Identify productive vs. unproductive discussion patterns
- **Decision Making**: Track how decisions are reached and communicated
- **Collaboration Quality**: Assess team interaction dynamics and knowledge sharing
- **Action Items**: Extract and categorize tasks, deadlines, and responsibilities

### Professional Insights
- **Project Progress**: Monitor milestone discussions and blockers
- **Team Dynamics**: Analyze leadership patterns, participation levels, and conflict resolution
- **Knowledge Transfer**: Identify learning opportunities and expertise sharing
- **Process Improvement**: Spot inefficiencies and suggest optimizations

### Key Metrics to Extract
- **Participation**: Who contributes most/least to discussions
- **Sentiment**: Overall team morale and engagement levels
- **Productivity Indicators**: Action items generated, decisions made, problems solved
- **Communication Clarity**: Identify unclear communications or misunderstandings

## Output Format

The analysis should be structured as a Python dictionary following this schema:

```python
class ConversationAnalysisResponse(BaseModel):
    summary: str
    users_feedback: dict[str, UserFeedback]

class UserFeedback(BaseModel):
    summary: str
    emoji: str
```

## Analysis Guidelines

- **Professional Tone**: Maintain objectivity and constructive feedback
- **Actionable Insights**: Focus on practical improvements and clear outcomes
- **Confidentiality**: Respect workplace privacy and professional boundaries
- **Constructive Feedback**: Highlight strengths and areas for improvement diplomatically

Here are the users and messages of the work conversation:
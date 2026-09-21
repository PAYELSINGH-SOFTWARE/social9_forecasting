from .gemini_service import generate_ai_text


def generate_content_calendar(
    platform: str,
    duration_days: int,
    topic: str
) -> str:

    prompt = f"""
You are a social media content strategist.

Create a {duration_days}-day content calendar.

Platform: {platform}
Topic: {topic}

For each day, provide:
1. Day number
2. Content topic
3. Content type
4. Engaging caption
5. Relevant hashtags
6. Suggested posting time

Return the calendar in clear JSON format.
Do not include markdown code blocks.
"""

    return generate_ai_text(prompt)
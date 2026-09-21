from src.gemini_service import generate_ai_text


prompt = "Write a short social media caption about AI forecasting."

result = generate_ai_text(prompt)

print("\nGemini Response:")
print(result)

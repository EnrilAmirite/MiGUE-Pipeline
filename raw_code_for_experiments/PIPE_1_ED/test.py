from openai import OpenAI

client = OpenAI(
    api_key=".......................",
    base_url="https://api.holdai.top/v1/chat/completions"
)

resp = client.responses.create(
    model="gpt-4.1-mini",
    input="Hello!"
)

print(resp.output_text)
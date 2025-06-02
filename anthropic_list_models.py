import anthropic
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

models = client.models.list(limit=20)

for model in models:
    print(model)

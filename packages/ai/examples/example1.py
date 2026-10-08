import asyncio

from pi_ai import create_models, SimpleStreamOptions
from pi_ai.providers.all import all_providers
from pi_ai.types import Context, UserMessage


async def main():
    models = create_models()
    for provider in all_providers():
        models.set_provider(provider)

    model = models.get_model("deepseek", "deepseek-flash")
    message = await models.complete_simple(
        model,
        Context(messages=[UserMessage(content="hello", timestamp=0)]),
        SimpleStreamOptions(api_key="sk-120a2914135c4db7a4b3708a177a8cbb"),
    )
    print(message.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())

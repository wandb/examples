"""One key, many models: list them, switch with MODEL_ID, and tune response settings."""

import weave

from shared.config import inference_client, model_id, require_api_key, weave_project

ANNOUNCER_PROMPT = "You are the Game Night Agent's announcer. Keep replies to one short sentence."


def main() -> None:
    require_api_key()
    weave.init(weave_project())
    client = inference_client()

    models = client.models.list()
    print(f"{len(models.data)} models available through one API key; using {model_id()}")

    response = client.chat.completions.create(
        model=model_id(),
        messages=[
            {"role": "system", "content": ANNOUNCER_PROMPT},
            {"role": "user", "content": "Announce that game night starts in ten minutes."},
        ],
        temperature=0.2,
        max_tokens=60,
    )
    print(response.choices[0].message.content)

    usage = response.usage
    if usage is None:
        print("usage: not reported")
    else:
        print(f"usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens")


if __name__ == "__main__":
    main()

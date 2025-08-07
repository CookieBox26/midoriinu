import json
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    DeveloperContent,
)
from vllm import LLM, SamplingParams


def main():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
        Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Always respond in riddles"),
        ),
        Message.from_role_and_content(Role.USER, "What is the weather like in SF?"),
    ])
    prefill_ids = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    stop_token_ids = encoding.stop_tokens_for_assistant_actions()

    llm = LLM(
        model="openai/gpt-oss-120b",
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        max_tokens=128,
        temperature=1,
        stop_token_ids=stop_token_ids,
    )
    outputs = llm.generate(
        prompt_token_ids=[prefill_ids],   # batch of size 1
        sampling_params=sampling,
    )

    gen = outputs[0].outputs[0]
    text = gen.text
    output_tokens = gen.token_ids
    entries = encoding.parse_messages_from_completion_tokens(output_tokens, Role.ASSISTANT)
    for message in entries:
        print(f'{json.dumps(message.to_dict())}')


if __name__ == '__main__':
    main()

from vllm import LLM

llm = LLM('openai/gpt-oss-120b', tensor_parallel_size=4)
resp = llm.generate('リーマン予想について教えてください。')
print(resp[0].outputs[0].text)

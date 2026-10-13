from agent.llm.deepseek.deepseek_client import DeepseekClient


def test_deepseek_client():
    message1 = "hello world"
    client = DeepseekClient(api_key="sk-")
    x = client.chat(human_message=message1, thinking="disabled")
    print(x)
    message2 = 'please tell 7+5 is what'
    y = client.chat(human_message=message2, thinking="disabled")
    print(y)


def test_deepseek_stream():
    client = DeepseekClient(api_key="sk-")
    message = "please introduce Bang Liu, the first emperor of HAN dynasty"
    print("stream output:")
    for token in client.stream(human_message=message, thinking="disabled"):
        print(token, end="", flush=True)
    print()
    print(f"history length: {len(client.history)}")

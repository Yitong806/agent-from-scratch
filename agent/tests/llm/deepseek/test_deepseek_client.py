from agent.llm.deepseek.deepseek_client import DeepseekClient


def test_deepseek_client():
    message1 = "hello world"
    client = DeepseekClient(api_key="sk-")
    x = client.chat(human_message=message1, thinking="disabled")
    print(x)
    message2 = 'please tell 7+5 is what'
    y = client.chat(human_message=message2, thinking="disabled")
    print(y)
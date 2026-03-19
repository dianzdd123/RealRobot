from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print("✅ OpenAI key loaded:", os.getenv("OPENAI_API_KEY") is not None)


functions = [
    {
        "name": "move_object",
        "description": "Move an object to a specified position.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "The name of the object to move, usually a fruit, like 'banana'."
                },
                "target_bin": {
                    "type": "string",
                    "description": "The target place, like 'yellow_bin'."
                }
            },
            "required": ["object_name", "target_bin"]
        }
    }
]

# 对话历史
messages = []

print("💬 开始实时对话（输入 q 退出）")

while True:
    user_input = input("你：")

    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 再见！")
        break

    # 添加到对话历史
    messages.append({"role": "user", "content": user_input})

    # 发送给 OpenAI
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto",
    )

    msg = response.choices[0].message
    messages.append(msg)  # 加入 assistant 响应，保留上下文

    # 处理函数调用
    if msg.function_call:
        func_name = msg.function_call.name
        arguments = json.loads(msg.function_call.arguments)
        print(f"\n🧠 ChatGPT识别函数调用：{func_name}")
        print(f"📦 参数：{arguments}")

        # 可选：实际调用控制函数（你来实现）
        # move_object(arguments["object_name"], arguments["target_bin"])
    else:
        print("🗣️ 回复：", msg.content)
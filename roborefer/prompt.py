def generate_simple_roborefer_prompt(user_text, output_path="roborefer_prompt.txt"):
    """
    从用户指令中提取 primary（抓取目标）和 target area（放置区域），
    并生成 RoboRefer 用的 prompt，保存到文件。
    """
    from openai import OpenAI
    client = OpenAI()

    # LLM 解析为两个关键信息
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个指令解析器。"
                "从用户输入中提取两个核心字段：\n"
                "1. primary: 需要抓取的主要物体（含修饰词，例如'最远处的香蕉'）\n"
                "2. target: 物体要放置的具体目标点或区域（含修饰词，例如'右边的柠檬和玉米之间的空地'）\n"
                "输出严格 JSON 格式：\n"
                "{\n"
                "  \"primary\": \"...\",\n"
                "  \"target\": \"...\"\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    parsed = resp.choices[0].message.content.strip()
    import json
    try:
        slots = json.loads(parsed)
    except json.JSONDecodeError:
        raise ValueError(f"LLM output 不是合法 JSON: {parsed}")

    # 生成 RoboRefer prompt
    roborefer_prompt = (
        f"Find the object \"{slots['primary']}\" and the target area \"{slots['target']}\" in the image. "
        "Return their coordinates in normalized [0,1] format as:\n"
        "{\n"
        "  \"primary\": [x_primary, y_primary],\n"
        "  \"target\": [x_target, y_target]\n"
        "}"
    )

    # 保存到文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(roborefer_prompt.strip())

    print(f"[OK] RoboRefer prompt 已保存到: {output_path}")
    return roborefer_prompt


if __name__ == "__main__":
    user_cmd = "移动最远处的香蕉到右边的柠檬和玉米之间的空地"
    prompt = generate_simple_roborefer_prompt(user_cmd)
    print("\n生成的 Prompt:\n", prompt)

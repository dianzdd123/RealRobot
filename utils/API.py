import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import re

# 初始化 API
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def subtasks_planner(user_input):
    # LLM 理解指令并输出任务步骤（自然语言描述）
    messages = [
    {
        "role": "system",
        "content": (
            "You are a robot assistant that understands and decomposes natural language instructions "
            "into a series of clear, structured subtasks.\n\n"
            "Your goal is to interpret the user's command and output a step-by-step plan, where each step describes:\n"
            "- what action to perform (e.g., move, pick up, place, rotate)\n"
            "- which object is being manipulated\n"
            "- what the goal or target is (e.g., place location, reference object)\n"
            "- any spatial relations or constraints (e.g., 'to the right of the apple')\n\n"
            "The output should be in plain text, without JSON, bullet points, or Markdown. Each step should be written as a clear sentence and in different linemov.\n\n"
            "Important: The subtask plan is intended to be used as the **foundation for robot trajectory generation**, so your steps must be precise and executable.\n\n"
            "Also, if a task involves approaching an object, the optimal method is usually to:\n"
            "- First move to a position directly above the object\n"
            "- Then descend vertically for grasping\n"
            "The same logic applies when placing an object near a target.\n\n"
            "Use this structured and pragmatic logic to output the most reliable subtask plan."
        )
    },
    {
        "role": "user",
        "content": user_input
    }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    print("\n Subtask Plan:")
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def trajectory_plan(subtask_txt_path="llm_subtasks.txt", grasp_json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    """
    基于任务拆解和抓取信息，调用 LLM 生成自然语言描述轨迹，并提取关键字段保存为 JSON 文件。

    返回:
        llm_output (str): LLM 原始自然语言轨迹描述
        parsed_steps (list): 结构化轨迹数据
    """
    # 读取任务描述
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    # 读取抓取信息
    with open(grasp_json_path, "r") as f:
        grasp_data = json.load(f)

    # 提问 LLM
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot trajectory planner. Based on the user's task breakdown and the sensed object information, "
                "generate a detailed motion plan for the robot arm to follow.\n"
                "Each step must include the following fields in plain text format:\n\n"
                "Step N:\n"
                "- Target 3D position: (x, y, z)\n"
                "- Target orientation: (x, y, z, w)\n"
                "- Gripper action: open / close / maintain grip\n"
                "- Gripper value: float% (e.g., 100.0%)\n"
                "- Description: What this step does\n\n"
                "Use this exact format, without Markdown symbols like ** or #, and keep each field on its own line.\n"
                "Make sure to use consistent punctuation and spacing so the output can be parsed with regular expressions."
            )
        },
        {
            "role": "user",
            "content": f"Task description:\n{subtask_text.strip()}\n\nSensed grasp data:\n{json.dumps(grasp_data, indent=2)}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    llm_output = response.choices[0].message.content
    print("\n LLM Trajectory Plan:\n")
    print(llm_output)

    # 保存 LLM 输出为文本
    # with open("trajectory_plan_raw.txt", "w") as f:
    #     f.write(llm_output)

    # 提取结构化字段为 JSON（用正则解析）
    steps = []
    pattern = re.compile(
        r"- Target 3D position:\s*\(([^)]+)\)\s*"
        r"- Target orientation:\s*\(([^)]+)\)\s*"
        r"- Gripper action:\s*(.*?)\s*"
        r"- Gripper value:\s*(\d*\.?\d+)%\s*",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(llm_output)
    print(f"✅ Found {len(matches)} steps.")

    for i, match in enumerate(matches):
        pos_str, quat_str, action, value = match
        pos = [float(x.strip()) for x in pos_str.split(",")]
        quat = [float(x.strip()) for x in quat_str.split(",")]
        value = float(value) / 100 if value else 0.0
        steps.append({
            "position": pos,
            "quaternion": quat,
            "gripper_action": action.lower(),
            "gripper_value": value
        })
    # print("========== REGEX DEBUG ==========")
    # for m in matches:
    #     print(m)
    # print("=================================")
    # 保存 JSON 文件
    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"\n✅ Trajectory steps saved to: {save_json_path}")

    return llm_output, steps

def trajectory_plan_roborefer(subtask_txt_path="llm_subtasks.txt", json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    # Load subtask description
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    # Load object information
    with open(json_path, "r") as f:
        grasp_data = json.load(f)

    # 精简信息，仅保留目标名和3D世界坐标
    simplified_data = []
    for item in grasp_data:
        if "3d_world" in item:
            simplified_data.append({
                "name": item["name"],
                "3d_world": item["3d_world"]
            })

    # 创建LLM prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot trajectory planner. Based on the user's task breakdown and the sensed object information, "
                "generate a detailed motion plan for the robot arm to follow.\n"
                "Each step must follow the format below:\n\n"
                "Step N:\n"
                " position: (x, y, z)\n"
                " gripper: 0 or 850\n"
                "0 means gripper closed; 850 means gripper open.\n"
                "Use this exact format (no Markdown). All units are in millimeters (mm)."
            )
        },
        {
            "role": "user",
            "content": (
                f"Task description:\n{subtask_text.strip()}\n\n"
                f"Sensed object data:\n{json.dumps(simplified_data, indent=2)}"
            )
        }
    ]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    llm_output = response.choices[0].message.content
    print("\nLLM Trajectory Plan:\n")
    print(llm_output)

    # 解析 LLM 输出（去掉 yaw 匹配）
    pattern = re.compile(
        r"position:\s*\(([^)]+)\)\s*"
        r"gripper:\s*([^\n]+)?"
    )

    matches = pattern.findall(llm_output)
    print(f"Found {len(matches)} steps.")

    steps = []
    for i, (pos_str, gripper_str) in enumerate(matches):
        try:
            pos = [float(x.strip()) for x in pos_str.split(",")] if pos_str else None
            gripper = int(gripper_str.strip()) if gripper_str else 850

            if pos:
                steps.append({
                    "position": pos,
                    "gripper": gripper
                })
            else:
                print(f"[WARN] Step {i+1} missing position.")

        except Exception as e:
            print(f"[ERROR] Failed to parse Step {i+1}: {e}")
            continue

    # 保存轨迹
    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"Trajectory steps saved to: {save_json_path}")

    return llm_output, steps

def trajectory_plan_on_real(subtask_txt_path="llm_subtasks.txt", json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    # Load subtask description
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    # Load grasp data from perception
    with open(json_path, "r") as f:
        grasp_data = json.load(f)

    # Call LLM for trajectory planning
    messages = [
        {
            "role": "system",
            "content": (
                "You are a robot trajectory planner. Based on the user's task breakdown and the sensed object information, "
                "generate a detailed motion plan for the robot arm to follow.\n"
                "About relative position, consider world frame x-axis positive direction is nearest user and y-axis positive direction is right hand"
                "Each step must include the following fields in plain text format:\n\n"
                "Step N:\n"
                " position: (x, y, z)\n"
                " yaw: n\n"
                " gripper: 0 or 850\n"
                " 0 means gripper close totally and 850 means gripper open"
                "Use this exact format, without Markdown symbols like ** or #.\n"
                "Units are in millimeters (mm) by default.\n"
                "If any field is missing, the system will use defaults (roll=180).\n"
                "Ensure that the motion trajectory covers a sufficiently wide range in 3D space, "
                "so that the robot arm demonstrates meaningful movement between steps (avoid tiny or redundant displacements).\n"
            )
        },
        {
            "role": "user",
            "content": f"Task description:\n{subtask_text.strip()}\n\nSensed grasp data:\n{json.dumps(grasp_data, indent=2)}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    llm_output = response.choices[0].message.content
    print("\nLLM Trajectory Plan:\n")
    print(llm_output)

    pattern = re.compile(
        r"position:\s*\(([^)]+)\)\s*"
        r"yaw:\s*([^\n]+)?\s*"
        r"gripper:\s*([^\n]+)?"
    )


    matches = pattern.findall(llm_output)
    print(f"Found {len(matches)} steps.")

    steps = []
    for i, (pos_str, yaw_str, gripper_str) in enumerate(matches):
        try:
            if pos_str:
                pos = [float(x.strip()) for x in pos_str.split(",")]
            else:
                print(f"[WARN] Step {i+1} missing position, skipping.")
                continue

            yaw = float(yaw_str.strip()) if yaw_str else 0.0
            gripper = int(gripper_str.strip()) if gripper_str else 850  # 默认张开

            steps.append({
                "position": pos,
                "yaw": yaw,
                "gripper": gripper
            })

        except Exception as e:
            print(f"[ERROR] Failed to parse Step {i+1}: {e}")
            continue


    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"Trajectory steps saved to: {save_json_path}")

    return llm_output, steps

def _extract_json_block(text: str) -> str:
    """从模型输出中提取 JSON 数组部分"""
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        return text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        return match.group(0)
    raise ValueError(f"Could not find JSON array in: {text[:200]}")

def roborefer_prompts(user_text: str) -> list:
    """
    Parse user command into multiple 'object/area of interest' entries
    and generate separate RoboRefer prompts for each.
    """
    # Step 1: Prompt for LLM
    system_instructions = (
        "You need to understand the user's instruction and extract the parts of the command "
        "that describe specific objects or areas of interest.\n"
        "For example:\n"
        '  Input: "Grab the apple at the far right of the table"\n'
        '  → Output: ["the apple at the far right of the table"]\n'
        '  Input: "Move the farthest banana to the empty space between the corn and the lemon"\n'
        '  → Output: ["the farthest banana", "the empty space between the corn and the lemon"]\n\n'
        "Return ONLY a JSON array of strings, in the same order they appear in the task, without any extra text."
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_text}
    ]

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0
    )
    parsed_raw = resp.choices[0].message.content or ""
    
    try:
        json_block = _extract_json_block(parsed_raw)
        slot_list = json.loads(json_block)
        if not isinstance(slot_list, list) or not slot_list:
            raise ValueError("Parsed result is not a non-empty list")
    except Exception as e:
        raise ValueError(f"Failed to parse object list: {e}\nLLM Output: {parsed_raw}")

    # Step 2: Generate RoboRefer prompts
    prompts = []
    for slot in slot_list:
        slot = slot.strip()
        rr_prompt = f'Find the object or area "{slot}" in the image.'
        prompts.append(rr_prompt)

    # Print extracted targets and prompts
    print("\n[Extracted Targets]")
    for t in slot_list:
        print("-", t)
    print("\n[Generated Prompts]")
    for p in prompts:
        print("-", p)

    return prompts

if __name__ == "__main__":
    # user_input = input("Enter your robot instruction: ")
    # task_description = subtasks_planner(user_input)
    # # Save parsed sub-tasks
    # with open("llm_subtasks.txt", "w") as f:
    #     f.write(task_description)
    # trajectory_plan_on_real2(subtask_txt_path="llm_subtasks.txt", json_path="imgs/img_from_camera.json", save_json_path="trajectory_plan.json")
    user_cmd = "grasp farest banana to corn left"
    prompt = roborefer_prompts(user_cmd)
import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# Initialize API client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def subtasks_planner(user_input):
    """Decomposes natural language instructions into a series of structured robotic subtasks."""
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
            "The output should be in plain text, without JSON, bullet points, or Markdown. Each step should be written as a clear sentence on different lines.\n\n"
            "Important: The subtask plan is intended to be used as the foundation for robot trajectory generation, so your steps must be precise and executable.\n\n"
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

    print("\n[INFO] Generated Subtask Plan:")
    print(response.choices[0].message.content)

    return response.choices[0].message.content

def trajectory_plan(subtask_txt_path="llm_subtasks.txt", grasp_json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    """Generates a trajectory plan and parses fields into a JSON file using regular expressions."""
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    with open(grasp_json_path, "r") as f:
        grasp_data = json.load(f)

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
    print("\n[INFO] LLM Trajectory Plan generated.")

    # Regex parsing for structured data extraction
    steps = []
    pattern = re.compile(
        r"- Target 3D position:\s*\(([^)]+)\)\s*"
        r"- Target orientation:\s*\(([^)]+)\)\s*"
        r"- Gripper action:\s*(.*?)\s*"
        r"- Gripper value:\s*(\d*\.?\d+)%\s*",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(llm_output)
    print(f"[INFO] Found {len(matches)} trajectory steps.")

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

    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"[INFO] Trajectory steps saved to: {save_json_path}")

    return llm_output, steps

def trajectory_plan_roborefer(subtask_txt_path="llm_subtasks.txt", json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    """Generates a simplified trajectory plan for RoboRefer targets."""
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    with open(json_path, "r") as f:
        grasp_data = json.load(f)

    # Simplify data to include only names and 3D world coordinates
    simplified_data = []
    for item in grasp_data:
        if "3d_world" in item:
            simplified_data.append({
                "name": item["name"],
                "3d_world": item["3d_world"]
            })

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

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.4
    )

    llm_output = response.choices[0].message.content
    print("\n[INFO] LLM RoboRefer Trajectory Plan generated.")

    pattern = re.compile(
        r"position:\s*\(([^)]+)\)\s*"
        r"gripper:\s*([^\n]+)?"
    )

    matches = pattern.findall(llm_output)
    steps = []
    for i, (pos_str, gripper_str) in enumerate(matches):
        try:
            pos = [float(x.strip()) for x in pos_str.split(",")] if pos_str else None
            gripper = int(gripper_str.strip()) if gripper_str else 850

            if pos:
                steps.append({"position": pos, "gripper": gripper})
            else:
                print(f"[WARN] Step {i+1} missing position coordinates.")

        except Exception as e:
            print(f"[ERROR] Failed to parse Step {i+1}: {e}")
            continue

    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"[INFO] Trajectory steps saved to: {save_json_path}")

    return llm_output, steps

def trajectory_plan_on_real(subtask_txt_path="llm_subtasks.txt", json_path="grasp_infos.json", save_json_path="trajectory_plan.json"):
    """Generates advanced trajectory plans including occlusion handling for real-robot execution."""
    with open(subtask_txt_path, "r") as f:
        subtask_text = f.read()

    with open(json_path, "r") as f:
        grasp_data = json.load(f)

    messages = [
        {
        "role": "system",
        "content": (
            "You are a robot trajectory planner. Based on the user's task breakdown and the sensed object information, "
            "generate a detailed motion plan for the robot arm to follow.\n"
            "About relative position, consider world frame x-axis positive direction is nearest user and y-axis positive direction is right hand.\n\n"
            "Occlusion handling:\n"
            "- The scene may include occlusions stored in the field \"occlusions\" (e.g., which objects block a target).\n"
            "- Before attempting to grasp or place the primary target, you must first remove ALL objects that occlude it.\n"
            "- Always move occluding objects to the fixed temporary placement area at position (-250, -440, 300).\n"
            "- After clearing all occluders, resume the main task with the target object.\n"
            "- If multiple occluders exist, remove them in a reasonable order (e.g., top-most/closest first) to minimize re-occlusion.\n\n"
            "Output format (plain text only, no Markdown):\n"
            "Each step must include exactly these fields:\n"
            "Step N:\n"
            " position: (x, y, z)\n"
            " yaw: n\n"
            " gripper: 0 or 850\n"
            "Where gripper=0 means fully closed, and gripper=850 means fully open.\n"
            "Use this exact format, without Markdown symbols like ** or #. Units are millimeters (mm).\n"
            "If any orientation field is missing, the system will use defaults (roll=180).\n\n"
            "Motion quality requirements:\n"
            "- Ensure the trajectory covers a sufficiently wide range in 3D space; avoid tiny or redundant displacements.\n"
            "- Use a safe approach: move above the target at a safe height, descend vertically to grasp/place."
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
    print("\n[INFO] LLM Advanced Trajectory Plan generated.")

    pattern = re.compile(
        r"position:\s*\(([^)]+)\)\s*"
        r"yaw:\s*([^\n]+)?\s*"
        r"gripper:\s*([^\n]+)?"
    )

    matches = pattern.findall(llm_output)
    steps = []
    for i, (pos_str, yaw_str, gripper_str) in enumerate(matches):
        try:
            if pos_str:
                pos = [float(x.strip()) for x in pos_str.split(",")]
            else:
                print(f"[WARN] Step {i+1} missing position, skipping.")
                continue

            yaw = float(yaw_str.strip()) if yaw_str else 0.0
            gripper = int(gripper_str.strip()) if gripper_str else 850

            steps.append({"position": pos, "yaw": yaw, "gripper": gripper})

        except Exception as e:
            print(f"[ERROR] Failed to parse Step {i+1}: {e}")
            continue

    with open(save_json_path, "w") as f:
        json.dump(steps, f, indent=2)
    print(f"[INFO] Trajectory steps saved to: {save_json_path}")

    return llm_output, steps

def _extract_json_block(text: str) -> str:
    """Extracts a JSON array block from the raw LLM output text."""
    text = text.strip()
    if text.startswith('[') and text.endswith(']'):
        return text
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        return match.group(0)
    raise ValueError(f"Could not find JSON array in output.")

def roborefer_prompts(user_text: str) -> list:
    """Parses user commands into multiple object-of-interest entries for RoboRefer."""
    system_instructions = (
        "You need to understand the user's instruction and extract the parts of the command "
        "that describe specific objects or areas of interest.\n"
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
    except Exception as e:
        raise ValueError(f"Failed to parse object list: {e}")

    prompts = []
    for slot in slot_list:
        rr_prompt = f'Find the object or area "{slot.strip()}" in the image.'
        prompts.append(rr_prompt)

    print("\n[INFO] Extracted Targets for RoboRefer:")
    for t in slot_list: print("-", t)

    return prompts

if __name__ == "__main__":
    # Example execution for validation
    user_cmd = "grasp farest banana to corn left"
    prompt = roborefer_prompts(user_cmd)
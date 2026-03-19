# test_rr_one_shot_parse_and_place.py
# 在 torch 环境运行；确保 roborefer 的 /query 服务已启动
import base64, json, ast, re, cv2, requests
from pathlib import Path

URL = "http://127.0.0.1:25547/query"
IMAGE_PATH = "imgs/img_from_camera.png"
OUT_IMG = "imgs/rr_parse_and_place.jpg"
ENABLE_DEPTH = 1

# 用户原始指令
USER_TEXT = "move the banana to the closest lemon's left"

# f-string 中 JSON 花括号需要双写 {{ }}
PROMPT_TEMPLATE = f"""
You are a vision-language grounding model. Read the user instruction and operate on the given image.

User instruction : "{USER_TEXT}"

First, parse it into slots (Chinese or English is fine):
- primary: the main object to manipulate (category name only, e.g., "banana")
- reference: the reference object/category (e.g., "lemon")
- relation: how to choose the reference (e.g., "nearest_to_primary")
- placement: the target placement rule relative to the reference (e.g., "right_of_reference")
- require_free_area: true if the target must be a free area on the table without overlapping objects

Then, in the image:
- Locate the primary object center coordinate.
- Locate ONE valid target placement point that satisfies the placement rule (must be a free area if require_free_area is true).

OUTPUT STRICTLY AS PURE JSON ONLY (no extra text), normalized coordinates in [0,1]:
{{
  "slots": {{
    "primary": "...",
    "reference": "...",
    "relation": "...",
    "placement": "...",
    "require_free_area": true
  }},
  "coords": {{
    "primary": [x_primary, y_primary],
    "target":  [x_target,  y_target]
  }}
}}
"""

def b64_of_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ask_rr(image_path, prompt, enable_depth=1):
    payload = {
        "image_url": [b64_of_image(image_path)],
        "depth_url": [],
        "enable_depth": int(enable_depth),
        "text": prompt
    }
    r = requests.post(URL, headers={"Content-Type":"application/json"}, data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    return r.json()

def extract_json(s):
    # 期望 answer 是纯 JSON；若有少量前后缀，尝试用正则截取第一个 {...}
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise

def draw_points(img_path, primary_xy, target_xy, save_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    def denorm(p): return (int(p[0]*w), int(p[1]*h))
    if primary_xy:
        x,y = denorm(primary_xy)
        cv2.circle(img,(x,y),10,(0,0,255),-1)
        cv2.putText(img,"primary",(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    if target_xy:
        x,y = denorm(target_xy)
        cv2.circle(img,(x,y),10,(0,200,0),-1)
        cv2.putText(img,"target",(x+8,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,0),2)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, img)

if __name__ == "__main__":
    prompt = PROMPT_TEMPLATE.format(user_text=USER_TEXT)
    resp = ask_rr(IMAGE_PATH, prompt, ENABLE_DEPTH)
    print("raw response:", resp)

    try:
        ans = extract_json(resp.get("answer",""))
        slots = ans.get("slots", {})
        coords = ans.get("coords", {})
        primary = coords.get("primary")
        target  = coords.get("target")
        print("parsed slots:", slots)
        print("coords (normalized):", coords)
    except Exception as e:
        print("Failed to parse JSON answer:", e)
        raise SystemExit(1)

    # 画点
    draw_points(IMAGE_PATH, primary, target, OUT_IMG)
    print(f"[OK] saved: {OUT_IMG}")

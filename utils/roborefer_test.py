import subprocess
import time
import json, ast, re
import base64
import requests
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.API import roborefer_prompts
from utils.rr_points import parse_points_any, denorm_points_to_pixels, extract_target_from_prompt

ROBOREF_ENV = "roborefer"
API_DIR = "/home/skyler/code/RoboRefer/API"
PORT = 25547
DEPTH_MODEL_PATH = "/home/skyler/code/RoboRefer/depth_anything_v2_vitl.pth"
VLM_MODEL_PATH = "Zhoues/RoboRefer-2B-SFT"
BASE = f"http://127.0.0.1:{PORT}"

def start_server():
    cmd = ["conda", "run", "-n", ROBOREF_ENV, "python", "api.py",
           "--port", str(PORT),
           "--depth_model_path", DEPTH_MODEL_PATH,
           "--vlm_model_path", VLM_MODEL_PATH]
    print("[INFO] 启动服务：", " ".join(cmd), f"(cwd={API_DIR})")
    return subprocess.Popen(cmd, cwd=API_DIR)

def wait_ready(timeout=300):
    print("[INFO] 等待服务就绪...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(BASE + "/", timeout=2)
            if r.status_code == 200 and "Hello, World!" in r.text:
                print("[INFO] 服务已就绪")
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False

def b64_of_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_query(image_path, prompt, enable_depth=1):
    payload = {
        "image_url": [b64_of_image(image_path)],
        "depth_url": [],
        "enable_depth": int(enable_depth),
        "text": prompt
    }
    headers = {"Content-Type": "application/json"}
    url = BASE + "/query"
    print("[INFO] POST", url)
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    return r.json()

def run_roborefer(image_path, prompts, server, verbose=False, save_path=None, enable_depth=1, return_name=True):
    """
    :param image_path: RGB图像路径
    :param prompts: 来自 roborefer_prompts 的 prompt 序列
    :param server: 通过 start_server() 启动的子进程对象
    :param verbose: 是否打印过程信息
    :param save_path: 如果设置，将保存可视化图像
    :param enable_depth: 是否启用深度估计
    :param return_name: 是否返回每个点的语义名称
    :return: 若 return_name=True，返回 (points_uv, descriptions)；否则只返回 points_uv
    """
    OUTPUT_SUFFIX = 'Return ONLY valid JSON with no extra text: {"points":[[x1,y1],[x2,y2],...]} All coordinates must be normalized to [0,1].'

    if not wait_ready():
        raise RuntimeError("服务启动超时")

    points_uv = []
    descriptions = []

    for i, base_p in enumerate(prompts, start=1):
        full_prompt = f"{base_p} {OUTPUT_SUFFIX}"
        if verbose:
            print(f"[Prompt #{i}] {full_prompt}")

        resp = call_query(image_path, full_prompt, enable_depth)
        answer = resp.get("answer", "")
        pts_norm = parse_points_any(answer)

        if not pts_norm:
            points_uv.append((None, None))
            descriptions.append(base_p)
            continue

        pts_px = denorm_points_to_pixels(image_path, [pts_norm[0]])
        points_uv.append(pts_px[0])
        descriptions.append(extract_target_from_prompt(base_p))

    if save_path:
        img = cv2.imread(image_path)
        for (x, y) in points_uv:
            if x is not None and y is not None:
                cv2.circle(img, (int(x), int(y)), 10, (255, 255, 255), -1)
                cv2.circle(img, (int(x), int(y)), 7, (50, 150, 255), -1)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, img)

    if return_name:
        return points_uv, descriptions
    else:
        return points_uv

if __name__ == "__main__":
    image_path = "imgs/2.png"
    user_cmd = "grasp the left banana"
    prompts = roborefer_prompts(user_cmd)
    
    server = start_server()
    points_uv, descriptions = run_roborefer(
        image_path=image_path,
        prompts=prompts,
        server=server,
        verbose=True,
        save_path="imgs/roborefer_point.jpg",
        return_name=True
    )
    server.terminate()
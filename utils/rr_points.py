import json, re, cv2
from typing import List, Tuple

def extract_target_from_prompt(prompt: str) -> str:
    """从 'Find the object or area "xxx" in the image.' 提取 xxx"""
    m = re.search(r'"([^"]+)"', prompt)
    return m.group(1) if m else prompt

def parse_points_any(answer_str: str) -> List[Tuple[float, float]]:
    """尽量稳健地解析 RoboRefer 的返回，得到归一化点 [(nx, ny), ...]"""
    import ast, json, re
    s = (answer_str or "").strip()

    # JSON 对象/数组
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "points" in obj:
            return [(float(x), float(y)) for x, y in obj["points"]]
        if isinstance(obj, list):
            out = []
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    out.append((float(it[0]), float(it[1])))
            if out: return out
    except Exception:
        pass

    # Python 字面量
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)):
            out = []
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) == 2:
                    out.append((float(it[0]), float(it[1])))
            if out: return out
    except Exception:
        pass

    # 正则兜底
    pairs = re.findall(r'[\(\[]\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*[\)\]]', s)
    return [(float(x), float(y)) for x, y in pairs]

def denorm_points_to_pixels(image_path: str, pts_norm: List[Tuple[float, float]]):
    """把归一化坐标转为像素坐标"""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    out = []
    for nx, ny in pts_norm:
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)
        out.append((int(nx*w), int(ny*h)))
    return out

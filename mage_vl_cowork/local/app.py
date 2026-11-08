"""Tkinter GUI for Mage-VL-Cowork running locally on Windows.

The model proposes `computer_use` (mouse/keyboard) and `bash` actions. Every
action is shown and must be approved by default, because the agent controls
THIS machine, not a sandbox. Slam the mouse into a screen corner (pyautogui
failsafe) or press Stop to abort.
"""
import argparse
import ctypes
import json
import re
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # pixel coords == pyautogui coords
except Exception:
    pass

import pyautogui
from PIL import ImageGrab

HERE = Path(__file__).parent
SYSTEM_PROMPT = (HERE / "system_prompt.txt").read_text(encoding="utf-8")
MAX_MODEL_PIXELS = 1280 * 720  # keep image tokens small for a 4 GB GPU
KEYMAP = {"ctrl": "ctrl", "control": "ctrl", "cmd": "win", "super": "win", "meta": "win",
          "return": "enter", "esc": "escape", "pgup": "pageup", "pgdn": "pagedown"}

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


# ----------------------------------------------------------------- model --
class Model:
    def __init__(self, path: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, llm_int8_enable_fp32_cpu_offload=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, quantization_config=quant, torch_dtype=torch.bfloat16,
            device_map="auto", max_memory={0: "3500MiB", "cpu": "12GiB"}).eval()

    def step(self, image, task: str, history: list[str]) -> str:
        text = f"Task: {task}"
        if history:
            text += "\n\nPrevious actions:\n" + "\n".join(f"{i+1}. {h}" for i, h in enumerate(history))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]},
        ]
        try:
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except TypeError:
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.model.device)
        inputs.pop("mm_token_type_ids", None)  # not accepted by this model's generate()
        if "patch_positions" not in inputs and "image_grid_thw" in inputs:
            # The vision tower needs per-patch [t, h, w] positions (raster order); the processor omits them.
            pos = []
            for t, h, w in inputs["image_grid_thw"].tolist():
                g = self.torch.stack(self.torch.meshgrid(
                    self.torch.arange(t), self.torch.arange(h), self.torch.arange(w), indexing="ij"), -1)
                pos.append(g.reshape(-1, 3))
            inputs["patch_positions"] = self.torch.cat(pos).to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.7, top_p=0.8, do_sample=True)
        return self.processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# --------------------------------------------------------------- parsing --
def parse(response: str):
    m = re.search(r"(?im)^\s*Action\s*:\s*(.+?)\s*$", response)
    summary = m.group(1).strip() if m else ""
    calls = []
    # closing tag optional: the model sometimes stops right after the JSON
    for mm in re.finditer(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|(?=<tool_call>)|$)", response, flags=re.DOTALL):
        try:
            obj = json.loads(mm.group(1))
            if isinstance(obj, dict):
                calls.append(obj)
        except json.JSONDecodeError:
            pass
    return summary, calls[:15]


def px(coord, w, h):
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    x = max(0.0, min(999.0, float(coord[0])))
    y = max(0.0, min(999.0, float(coord[1])))
    return round(x / 999 * (w - 1)), round(y / 999 * (h - 1))


def describe(call) -> str:
    return json.dumps(call.get("arguments", {}), ensure_ascii=False) if call.get("name") == "computer_use" \
        else "bash: " + str(call.get("arguments", {}).get("command", ""))


def execute(call, w, h) -> str:
    """Run one tool call. Returns text output ('' for GUI actions); 'DONE'/'FAIL' for terminate."""
    a = call.get("arguments") or {}
    if call.get("name") == "bash":
        r = subprocess.run(["powershell", "-NoProfile", "-Command", str(a.get("command", ""))],
                           capture_output=True, text=True, timeout=300)
        return (r.stdout + r.stderr).strip() or "(no output)"
    act = a.get("action")
    xy = px(a.get("coordinate"), w, h)
    if act in ("click", "double_click", "triple_click", "move") and xy:
        pyautogui.moveTo(*xy, duration=0.2)
        if act == "click":
            pyautogui.click(button=a.get("button", "left"))
        elif act == "double_click":
            pyautogui.doubleClick()
        elif act == "triple_click":
            pyautogui.click(clicks=3, interval=0.05)
    elif act == "drag":
        s, e = px(a.get("start"), w, h), px(a.get("end"), w, h)
        if s and e:
            pyautogui.moveTo(*s)
            pyautogui.dragTo(*e, duration=0.5, button="left")
    elif act == "type":
        import pyperclip
        pyperclip.copy(str(a.get("text", "")))
        pyautogui.hotkey("ctrl", "v")
    elif act == "keypress":
        keys = a.get("keys") or ([a["key"]] if a.get("key") else [])
        if isinstance(keys, str):
            keys = [k.strip() for k in keys.split("+")]
        keys = [KEYMAP.get(str(k).lower(), str(k).lower()) for k in keys]
        if len(keys) > 1:
            pyautogui.hotkey(*keys)
        elif keys:
            pyautogui.press(keys[0])
    elif act == "scroll":
        if xy:
            pyautogui.moveTo(*xy)
        sy, sx = float(a.get("scroll_y") or 0), float(a.get("scroll_x") or 0)
        if sy:
            pyautogui.scroll(int(sy / 999 * 10) or (1 if sy > 0 else -1))
        if sx:
            pyautogui.hscroll(int(sx / 999 * 10) or (1 if sx > 0 else -1))
    elif act == "wait":
        time.sleep(max(0.1, float(a.get("time") or 1)))
    elif act == "terminate":
        return "FAIL" if a.get("status") == "failure" else "DONE"
    return ""


# ------------------------------------------------------------------- GUI --
class App(tk.Tk):
    def __init__(self, model_path):
        super().__init__()
        self.title("Mage-VL-Cowork")
        self.geometry("620x520")
        self.model_path, self.model = model_path, None
        self.stop_flag = threading.Event()
        self.approve = tk.BooleanVar(value=True)
        self.max_steps = tk.IntVar(value=30)

        tk.Label(self, text="Task:").pack(anchor="w", padx=8, pady=(8, 0))
        self.task = tk.Text(self, height=4)
        self.task.pack(fill="x", padx=8)
        bar = tk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        self.run_btn = tk.Button(bar, text="Run", command=self.start, state="disabled")
        self.run_btn.pack(side="left")
        tk.Button(bar, text="Stop", command=self.stop_flag.set).pack(side="left", padx=4)
        tk.Checkbutton(bar, text="Approve each step (recommended)", variable=self.approve).pack(side="left", padx=10)
        tk.Label(bar, text="Max steps:").pack(side="left")
        tk.Spinbox(bar, from_=1, to=100, width=4, textvariable=self.max_steps).pack(side="left")
        self.log = scrolledtext.ScrolledText(self, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.say("Loading model (first load takes a few minutes)...")
        threading.Thread(target=self.load, daemon=True).start()

    def say(self, msg):
        def _w():
            self.log.configure(state="normal")
            self.log.insert("end", msg + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.after(0, _w)

    def load(self):
        try:
            self.model = Model(self.model_path)
            self.say("Model ready. Enter a task and press Run.")
            self.after(0, lambda: self.run_btn.configure(state="normal"))
        except Exception as e:
            self.say(f"Model load failed: {e!r}")

    def start(self):
        task = self.task.get("1.0", "end").strip()
        if not task:
            return
        self.stop_flag.clear()
        self.run_btn.configure(state="disabled")
        threading.Thread(target=self.loop, args=(task,), daemon=True).start()

    def ask(self, text) -> bool:
        res, ev = [], threading.Event()
        def _a():
            self.deiconify()
            res.append(messagebox.askyesno("Approve action?", text, parent=self))
            ev.set()
        self.after(0, _a)
        ev.wait()
        return res[0]

    def loop(self, task):
        history = []
        try:
            for n in range(1, self.max_steps.get() + 1):
                if self.stop_flag.is_set():
                    self.say("Stopped.")
                    break
                self.after(0, self.iconify)  # keep this window out of the screenshot
                time.sleep(0.8)
                shot = ImageGrab.grab().convert("RGB")
                w, h = shot.size
                scale = (MAX_MODEL_PIXELS / (w * h)) ** 0.5
                img = shot.resize((int(w * scale), int(h * scale))) if scale < 1 else shot
                self.say(f"\n--- Step {n}: thinking...")
                resp = self.model.step(img, task, history)
                summary, calls = parse(resp)
                self.say(f"Action: {summary or '(none)'}")
                if not calls:
                    self.say("No tool call produced; stopping.\n--- raw model output ---\n" + resp)
                    break
                desc = "\n".join(describe(c) for c in calls)
                self.say(desc)
                # bash always needs approval; GUI steps follow the checkbox
                needs = self.approve.get() or any(c.get("name") == "bash" for c in calls)
                if needs and not self.ask(f"{summary}\n\n{desc}"):
                    self.say("Rejected by user. Stopping.")
                    break
                self.after(0, self.iconify)
                time.sleep(0.5)
                outputs, finished = [], None
                for c in calls:
                    out = execute(c, w, h)
                    if out in ("DONE", "FAIL"):
                        finished = out
                        break
                    if out:
                        outputs.append(out)
                history.append(summary or desc[:80])
                if outputs:
                    self.say("Output:\n" + "\n".join(outputs))
                if finished:
                    self.say(f"Agent finished: {finished}")
                    break
                time.sleep(1.0)
            else:
                self.say("Reached max steps.")
        except pyautogui.FailSafeException:
            self.say("Aborted via mouse failsafe.")
        except Exception as e:
            self.say(f"Error: {e!r}")
        finally:
            self.after(0, self.deiconify)
            self.after(0, lambda: self.run_btn.configure(state="normal"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    App(ap.parse_args().model).mainloop()

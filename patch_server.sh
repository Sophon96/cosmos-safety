#!/bin/bash
# Patch server files to fix:
# 1. Recursive remote call bug in reason_server.py
# 2. Missing enable_thinking=True in reason.py (model skips chain-of-thought)
# 3. Add logging for raw responses in reason.py
# 4. Fix prompt.txt to not include literal <think> tags
#
# Usage: bash patch_server.sh

set -e

cd "$(dirname "$0")"

patch -p1 <<'PATCH'
diff --git a/prompt.txt b/prompt.txt
index d992e04..9407792 100644
--- a/prompt.txt
+++ b/prompt.txt
@@ -1,7 +1,5 @@
 The video shows a robot trying to pour liquid from one cup into another. Based on the robot's current movement and position, will the cup the gripper is holding be in the proper position on top of the other cup when it starts pouring?

-<think>
-Your reasoning here.
-</think>
+Think step by step about the robot's trajectory, gripper position, and alignment with the target cup before giving your answer.

-Your final answer: 1 if yes (on track), 0 if no (not on track). Reply with only the number.
\ No newline at end of file
+Your final answer: 1 if yes (on track), 0 if no (not on track). Reply with only the number after your reasoning.
\ No newline at end of file
diff --git a/reason.py b/reason.py
index 2c3153a..df4ea88 100644
--- a/reason.py
+++ b/reason.py
@@ -1,5 +1,6 @@
 # Unsloth must be imported before transformers for optimizations
 # from unsloth import FastVisionModel
+import logging
 import os
 import torch
 import transformers
@@ -7,6 +8,8 @@ import time
 from pathlib import Path
 from typing import Any

+logger = logging.getLogger(__name__)
+
 MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
 BINARY_PROMPT = (
     "Watch the video. Is the robot about to pour water from one cup into another? "
@@ -72,7 +75,9 @@ def _binary_check_remote(url: str, video_path: str | Path, fps: int) -> int:
         with httpx.Client(timeout=30.0) as client:
             r = client.post(f"{url}/binary_check", files=files, data=data)
             r.raise_for_status()
-            return int(r.json()["result"])
+            response = r.json()
+            logger.info(f"Binary check remote raw response: {response}")
+            return int(response["result"])


 def cosmos_binary_check(
@@ -138,6 +143,7 @@ def cosmos_binary_check(
     )

     text = (output_text[0] or "").strip().lower()
+    logger.info(f"Binary check raw response: {text}")
     # Parse 1/0 or yes/no (1 = about to pour)
     if "1" in text or text.startswith("yes"):
         return 1
@@ -163,7 +169,9 @@ def _full_reason_remote(
         with httpx.Client(timeout=120.0) as client:
             r = client.post(f"{url}/full_reason", files=files, data=data)
             r.raise_for_status()
-            return r.json().get("output", "")
+            response = r.json()
+            logger.info(f"Full reason remote raw response: {response}")
+            return response.get("output", "")


 def cosmos_full_reason(
@@ -221,6 +229,7 @@ def cosmos_full_reason(
         return_dict=True,
         return_tensors="pt",
         fps=fps,
+        enable_thinking=True,
     )
     inputs = inputs.to(model.device)

@@ -235,7 +244,9 @@ def cosmos_full_reason(
         clean_up_tokenization_spaces=False,
     )

-    return output_text[0] or ""
+    result = output_text[0] or ""
+    logger.info(f"Full reason local raw response: {result}")
+    return result


 # Standalone script behavior (original reason.py)
diff --git a/reason_server.py b/reason_server.py
index e4241e9..158ab83 100644
--- a/reason_server.py
+++ b/reason_server.py
@@ -6,15 +6,22 @@ Run on cloud: uvicorn reason_server:app --host 0.0.0.0 --port 8000
 Local client sets COSMOS_REMOTE_URL=http://<cloud>:8000 to use this server.
 """

+import os
 import tempfile
 from pathlib import Path

 from fastapi import FastAPI, File, Form, UploadFile

-from reason import cosmos_binary_check, cosmos_full_reason
+# Unset COSMOS_REMOTE_URL so the server always uses local inference
+os.environ.pop("COSMOS_REMOTE_URL", None)
+
+from reason import cosmos_binary_check, cosmos_full_reason, load_cosmos_model

 app = FastAPI(title="Cosmos Reason VLM", version="0.1.0")

+# Pre-load model at startup
+_model, _processor = load_cosmos_model()
+

 @app.post("/binary_check")
 async def binary_check(
@@ -30,7 +37,7 @@ async def binary_check(
         f.write(content)
         path = Path(f.name)
     try:
-        result = cosmos_binary_check(path, fps=fps)
+        result = cosmos_binary_check(path, model=_model, processor=_processor, fps=fps)
         return {"result": result}
     finally:
         path.unlink(missing_ok=True)
@@ -61,6 +68,8 @@ async def full_reason(
                 output = cosmos_full_reason(
                     path,
                     prompt_path=prompt_path,
+                    model=_model,
+                    processor=_processor,
                     fps=fps,
                     max_new_tokens=max_new_tokens,
                 )
@@ -70,6 +79,8 @@ async def full_reason(
             output = cosmos_full_reason(
                 path,
                 prompt_path="prompt.txt",
+                model=_model,
+                processor=_processor,
                 fps=fps,
                 max_new_tokens=max_new_tokens,
             )
PATCH

echo "Patched reason.py, reason_server.py, and prompt.txt successfully."
echo "Restart the server to apply changes."

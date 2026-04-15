"""
FastAPI-сервис для Куриного Анализатора.

Принимает изображение → запускает двухэтапный pipeline в отдельном потоке →
стримит прогресс через Server-Sent Events (SSE).
"""

import asyncio
import gc
import json
import os
import re
import tempfile
import threading
from queue import Queue

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Куриный Анализатор API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  Вспомогательные функции
# ─────────────────────────────────────────────


def detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def unload_model(model, device: str) -> None:
    import torch

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────
#  Pipeline (запускается в отдельном потоке)
# ─────────────────────────────────────────────


def run_pipeline(image_path: str, device: str, queue: Queue) -> None:
    """
    Двухэтапный pipeline анализа яйца.
    Прогресс отправляется в queue как dict-события.
    None в конце — sentinel (завершение стрима).
    """

    def emit(data: dict) -> None:
        queue.put(data)

    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        # ══════════════════════════════════════
        #  ЭТАП 1: Vision (Moondream2)
        # ══════════════════════════════════════
        emit(
            {
                "type": "stage",
                "stage": 1,
                "message": "Загрузка vision-модели (Moondream2)...",
            }
        )

        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            revision="2025-04-14",
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map={"": device},
        )

        image = Image.open(image_path).convert("RGB")
        settings = {"max_tokens": 512}

        emit({"type": "progress", "message": "Проверка наличия яйца на изображении..."})
        egg_check = (
            model.query(
                image,
                "Is there an egg or eggs visible in this image? Answer only YES or NO.",
                settings={"max_tokens": 10},
            )["answer"]
            .strip()
            .upper()
        )

        if not egg_check.startswith("YES"):
            unload_model(model, device)
            emit(
                {
                    "type": "result",
                    "data": {
                        "quality": "error",
                        "verdict_ru": "ЯЙЦО НЕ ОБНАРУЖЕНО",
                        "confidence": "high",
                        "defects_found": [],
                        "reasoning": "На изображении не найдено яйцо для анализа. Подайте фото яйца.",
                        "no_egg": True,
                    },
                }
            )
            return

        emit({"type": "progress", "message": "Анализ состояния скорлупы..."})
        shell_desc = model.query(
            image,
            "Describe the condition of this egg or eggs in detail. "
            "Focus on shell integrity: cracks, holes, chips, dirt, stains, "
            "deformation, discoloration, or any abnormalities.",
            settings=settings,
        )["answer"]

        emit({"type": "progress", "message": "Поиск дефектов и повреждений..."})
        defects_desc = model.query(
            image,
            "List every visible defect, damage, or abnormality you can find "
            "on this egg or eggs. If there are no defects, say 'No defects found'.",
            settings=settings,
        )["answer"]

        emit({"type": "progress", "message": "Общая оценка пригодности к продаже..."})
        assessment = model.query(
            image,
            "Would you pass this egg as suitable for sale in a store? "
            "Answer YES or NO and explain why briefly.",
            settings=settings,
        )["answer"]

        vision_report = (
            "=== VISUAL INSPECTION REPORT ===\n"
            f"1. SHELL CONDITION:\n{shell_desc}\n\n"
            f"2. DETECTED DEFECTS:\n{defects_desc}\n\n"
            f"3. SALE ASSESSMENT:\n{assessment}\n"
        )

        emit({"type": "progress", "message": "Выгрузка vision-модели из памяти..."})
        unload_model(model, device)

        # ══════════════════════════════════════
        #  ЭТАП 2: Reasoning (Qwen2.5-1.5B)
        # ══════════════════════════════════════
        emit(
            {
                "type": "stage",
                "stage": 2,
                "message": "Загрузка text-модели (Qwen2.5-1.5B)...",
            }
        )

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device},
        )

        system_prompt = (
            "You are a senior egg quality inspector at a poultry plant. "
            "You receive a visual inspection report from a camera system and must "
            "make the final quality decision.\n\n"
            "Rules:\n"
            "- quality = 'good' ONLY if the egg has NO defects at all\n"
            "- quality = 'bad' if ANY defect is found (cracks, holes, dirt, damage, etc.)\n"
            "- When in doubt, mark as 'bad' (safety first)\n\n"
            "Respond ONLY with valid JSON in this exact format, nothing else:\n"
            '{"quality": "good or bad", "confidence": "high/medium/low", '
            '"defects_found": ["short string defect 1", "short string defect 2"], '
            '"reasoning": "brief explanation of your decision"}\n\n'
            "IMPORTANT: defects_found must be a flat array of SHORT plain strings only. "
            'No objects, no nested keys. Example: ["crack on shell", "dirt stain"]'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Visual inspection report:\n\n{vision_report}\n\n"
                    "Make your quality decision. Return ONLY the JSON."
                ),
            },
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(device)

        emit({"type": "progress", "message": "Вынесение итогового вердикта..."})

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        raw = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

        emit({"type": "progress", "message": "Выгрузка text-модели из памяти..."})
        unload_model(model, device)

        # Парсинг JSON-ответа
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                data = json.loads(match.group(0))
                quality = data.get("quality", "bad")
                data["verdict_ru"] = "ГОДНОЕ" if quality == "good" else "БРАК"
                emit({"type": "result", "data": data})
                return
            except json.JSONDecodeError:
                pass

        # Fallback при ошибке парсинга
        emit(
            {
                "type": "result",
                "data": {
                    "quality": "bad",
                    "verdict_ru": "БРАК",
                    "confidence": "low",
                    "defects_found": ["parse_error"],
                    "reasoning": f"Не удалось разобрать ответ модели: {raw[:300]}",
                },
            }
        )

    except Exception as exc:
        emit({"type": "error", "message": str(exc)})
    finally:
        queue.put(None)  # sentinel — завершение стрима


# ─────────────────────────────────────────────
#  HTTP Endpoints
# ─────────────────────────────────────────────


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Принимает изображение, возвращает SSE-стрим с событиями прогресса и результатом.

    События:
      {"type": "stage",    "stage": 1, "message": "..."}
      {"type": "progress", "message": "..."}
      {"type": "result",   "data": {...}}
      {"type": "error",    "message": "..."}
    """
    suffix = os.path.splitext(file.filename or "image")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    device = detect_device()
    queue: Queue = Queue()

    thread = threading.Thread(
        target=run_pipeline,
        args=(tmp_path, device, queue),
        daemon=True,
    )
    thread.start()

    async def event_stream():
        try:
            while True:
                item = await asyncio.to_thread(queue.get)
                if item is None:
                    break
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "Куриный Анализатор"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

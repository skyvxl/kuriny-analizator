"""
Анализатор изображений — MVP (двухэтапный pipeline)
«Контроль качества продукции на птицефабрике при помощи компьютерного зрения»

Архитектура:
    Этап 1 — Vision-модель (Moondream2, 2B) анализирует фото,
             описывает что видит, выгружается из памяти.
    Этап 2 — Text LLM (Qwen2.5-1.5B-Instruct) получает описание,
             рассуждает, выносит вердикт и формирует JSON.

Использование:
    python analyze.py --image egg.jpg
    python analyze.py --image egg.jpg --device mps     # Mac
    python analyze.py --image egg.jpg --device cpu      # без GPU
"""

import argparse
import sys
import os
import time
import json
import gc


def detect_device() -> str:
    """Автоопределение лучшего доступного устройства."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🟢 NVIDIA GPU: {name} ({vram:.1f} GB VRAM)")
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("🟢 Apple Silicon (MPS)")
        return "mps"

    print("🟡 GPU не найден — CPU")
    return "cpu"


def unload_model(model, device: str):
    """Полностью выгружает модель из GPU/RAM."""
    import torch

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        vram_free = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"   Освобождено GPU, свободно {vram_free:.1f} GB VRAM")


# ═══════════════════════════════════════════════
#  ЭТАП 1: Vision-модель описывает изображение
# ═══════════════════════════════════════════════
def stage1_vision(image_path: str, device: str) -> str:
    """Moondream2 смотрит на изображение и подробно описывает."""
    import torch
    from transformers import AutoModelForCausalLM
    from PIL import Image

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print("\n" + "─" * 60)
    print("📸 ЭТАП 1: Визуальный анализ (Moondream2, 2B)")
    print("─" * 60)
    print("⏳ Загрузка vision-модели...")

    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-04-14",
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )

    image = Image.open(image_path).convert("RGB")
    settings = {"max_tokens": 512}

    # Несколько ракурсов анализа
    print("🔍 Описание общего состояния...")
    desc = model.query(
        image,
        "Describe the condition of this egg or eggs in detail. "
        "Focus on shell integrity: cracks, holes, chips, dirt, stains, "
        "deformation, discoloration, or any abnormalities.",
        settings=settings,
    )["answer"]

    print("🔍 Поиск дефектов...")
    defects = model.query(
        image,
        "List every visible defect, damage, or abnormality you can find "
        "on this egg or eggs. If there are no defects, say 'No defects found'.",
        settings=settings,
    )["answer"]

    print("🔍 Общая оценка...")
    assessment = model.query(
        image,
        "Would you pass this egg as suitable for sale in a store? "
        "Answer YES or NO and explain why briefly.",
        settings=settings,
    )["answer"]

    # Собираем полный отчёт vision-модели
    vision_report = (
        f"=== VISUAL INSPECTION REPORT ===\n"
        f"Image: {image_path}\n\n"
        f"1. SHELL CONDITION:\n{desc}\n\n"
        f"2. DETECTED DEFECTS:\n{defects}\n\n"
        f"3. SALE ASSESSMENT:\n{assessment}\n"
    )

    print(f"\n📋 Отчёт vision-модели:\n{vision_report}")

    # Выгружаем vision-модель
    print("🗑️  Выгрузка vision-модели из памяти...")
    unload_model(model, device)

    return vision_report


# ═══════════════════════════════════════════════
#  ЭТАП 2: Text LLM рассуждает и выносит вердикт
# ═══════════════════════════════════════════════
def stage2_reasoning(vision_report: str, device: str) -> str:
    """Qwen2.5-1.5B-Instruct анализирует отчёт и выдаёт итоговый JSON."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print("\n" + "─" * 60)
    print("🧠 ЭТАП 2: Анализ и решение (Qwen2.5-1.5B-Instruct)")
    print("─" * 60)
    print("⏳ Загрузка text-модели...")

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
        '"defects_found": ["list", "of", "defects"], '
        '"reasoning": "brief explanation of your decision"}'
    )

    user_message = (
        f"Here is the visual inspection report from the camera:\n\n"
        f"{vision_report}\n\n"
        f"Make your quality decision. Return ONLY the JSON."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    print("🔍 Анализ отчёта и вынесение решения...")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
        )

    # Декодируем только сгенерированную часть
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw_answer = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()

    # Выгружаем text-модель
    print("🗑️  Выгрузка text-модели из памяти...")
    unload_model(model, device)

    return format_final_result(raw_answer)


def format_final_result(raw: str) -> str:
    """Парсим JSON из ответа LLM, добавляем русский вердикт."""
    import re

    # Пробуем извлечь JSON
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            data = json.loads(match.group(0))
            quality = data.get("quality", "bad")
            data["verdict_ru"] = "✅ ГОДНОЕ" if quality == "good" else "❌ БРАК"
            return json.dumps(data, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    # Fallback
    return json.dumps({
        "quality": "bad",
        "verdict_ru": "❌ БРАК (не удалось распарсить ответ)",
        "confidence": "low",
        "defects_found": ["parse_error"],
        "reasoning": f"Raw model output: {raw[:500]}",
    }, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="🐔 Куриный Анализатор"
    )
    parser.add_argument(
        "--image", "-i", default="input.png",
        help="Путь к изображению (по умолчанию: input.png)"
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Устройство: auto, cuda, mps, cpu"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Файл не найден: {args.image}")
        sys.exit(1)

    device = detect_device() if args.device == "auto" else args.device

    print("=" * 60)
    print("🐔 КУРИНЫЙ АНАЛИЗАТОР")
    print("   Двухэтапный pipeline (Vision → LLM)")
    print("=" * 60)
    print(f"📄 Файл:        {args.image}")
    print(f"🖥️  Устройство:  {device}")
    print(f"📸 Этап 1:      Moondream2 (2B) — визуальный анализ")
    print(f"🧠 Этап 2:      Qwen2.5-1.5B — решение и JSON")

    start = time.time()

    # Этап 1: Vision
    vision_report = stage1_vision(args.image, device)

    # Этап 2: Reasoning
    final_json = stage2_reasoning(vision_report, device)

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("📝 ИТОГОВЫЙ ВЕРДИКТ:")
    print("=" * 60)
    print(final_json)
    print("─" * 60)
    print(f"⏱️  Общее время: {elapsed:.1f} сек.")
    print("=" * 60)


if __name__ == "__main__":
    main()
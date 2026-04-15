<script lang="ts">
	import type { AnalysisResult, ProgressLog } from '$lib/types';

	// ── State ────────────────────────────────────────────────
	let dragOver    = $state(false);
	let file        = $state<File | null>(null);
	let previewUrl  = $state<string | null>(null);
	let analyzing   = $state(false);
	let done        = $state(false);
	let stage       = $state(0);
	let logs        = $state<ProgressLog[]>([]);
	let result      = $state<AnalysisResult | null>(null);
	let errorMsg    = $state<string | null>(null);

	// ── Derived ──────────────────────────────────────────────
	let canAnalyze = $derived(!!file && !analyzing);

	// ── File handling ────────────────────────────────────────
	function setFile(f: File) {
		if (!f.type.startsWith('image/')) return;
		if (previewUrl) URL.revokeObjectURL(previewUrl);
		file       = f;
		previewUrl = URL.createObjectURL(f);
		result     = null;
		errorMsg   = null;
		logs       = [];
		stage      = 0;
		done       = false;
	}

	function onFileChange(e: Event) {
		const input = e.target as HTMLInputElement;
		const f = input.files?.[0];
		if (f) setFile(f);
	}

	function onDragOver(e: DragEvent) { e.preventDefault(); dragOver = true; }
	function onDragLeave(e: DragEvent) { e.preventDefault(); dragOver = false; }
	function onDrop(e: DragEvent) {
		e.preventDefault();
		dragOver = false;
		const f = e.dataTransfer?.files[0];
		if (f) setFile(f);
	}

	// ── Analysis ─────────────────────────────────────────────
	async function analyze() {
		if (!file || analyzing) return;

		analyzing = true;
		done      = false;
		result    = null;
		errorMsg  = null;
		logs      = [];
		stage     = 0;

		const fd = new FormData();
		fd.append('file', file);

		try {
			const res = await fetch('/api/analyze', { method: 'POST', body: fd });
			if (!res.body) throw new Error('Пустой ответ сервера');

			const reader  = res.body.getReader();
			const decoder = new TextDecoder();
			let   buffer  = '';

			while (true) {
				const { done: streamDone, value } = await reader.read();
				if (streamDone) break;

				buffer += decoder.decode(value, { stream: true });

				// SSE-события разделены "\n\n"
				const parts = buffer.split('\n\n');
				buffer = parts.pop() ?? '';

				for (const part of parts) {
					const line = part.split('\n').find(l => l.startsWith('data: '));
					if (!line) continue;
					try { handleEvent(JSON.parse(line.slice(6))); } catch { /* skip */ }
				}
			}
		} catch (err) {
			errorMsg = err instanceof Error ? err.message : 'Неизвестная ошибка';
		} finally {
			analyzing = false;
			done      = true;
		}
	}

	function handleEvent(ev: { type: string; stage?: number; message?: string; data?: AnalysisResult }) {
		if (ev.type === 'stage') {
			stage = ev.stage ?? stage;
			logs  = [...logs, { text: ev.message ?? '', kind: 'stage', stage: ev.stage }];
		} else if (ev.type === 'progress') {
			logs = [...logs, { text: ev.message ?? '', kind: 'progress' }];
		} else if (ev.type === 'result') {
			const data = ev.data ?? null;
			if (data && Array.isArray(data.defects_found)) {
				data.defects_found = data.defects_found.map((d: unknown) => {
					if (typeof d === 'string') return d;
					if (d && typeof d === 'object') {
						const o = d as Record<string, string>;
						if (o.type && o.description) return `${o.type}: ${o.description}`;
						if (o.type) return o.type;
						if (o.description) return o.description;
					}
					return JSON.stringify(d);
				});
			}
			result = data;
		} else if (ev.type === 'error') {
			errorMsg = ev.message ?? 'Ошибка анализа';
		}
	}

	function reset() {
		if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
		file = null; result = null; errorMsg = null;
		logs = []; stage = 0; analyzing = false; done = false;
	}

	const CONFIDENCE: Record<string, { label: string; cls: string }> = {
		high:   { label: 'Высокая',  cls: 'text-green-400' },
		medium: { label: 'Средняя',  cls: 'text-yellow-400' },
		low:    { label: 'Низкая',   cls: 'text-red-400' },
	};
</script>

<div class="min-h-screen bg-surface text-prose">

	<!-- Header -->
	<header class="border-b border-rim bg-card">
		<div class="mx-auto flex max-w-5xl items-center gap-3 px-6 py-4">
			<span class="text-3xl">🐔</span>
			<div>
				<h1 class="text-lg font-bold tracking-tight">Куриный Анализатор</h1>
				<p class="text-xs text-muted">Контроль качества · Компьютерное зрение · AI</p>
			</div>
			<div class="ml-auto flex items-center gap-2 rounded-full border border-rim bg-surface px-3 py-1.5">
				<span class="size-2 animate-pulse rounded-full bg-green-400"></span>
				<span class="text-xs text-muted">Vision → LLM Pipeline</span>
			</div>
		</div>
	</header>

	<main class="mx-auto max-w-5xl px-6 py-8">
		<div class="grid gap-6 lg:grid-cols-2">

			<!-- ── Левая колонка: загрузка ───────────────── -->
			<section class="flex flex-col gap-4">
				<h2 class="text-xs font-semibold uppercase tracking-widest text-muted">📤 Загрузка</h2>

				{#if !previewUrl}
					<!-- Drop zone -->
					<label
						class="group relative flex min-h-52 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed transition-colors
							{dragOver ? 'border-amber-400 bg-amber-400/5' : 'border-rim bg-card hover:border-amber-500/50 hover:bg-elevated'}"
						ondragover={onDragOver}
						ondragleave={onDragLeave}
						ondrop={onDrop}
					>
						<input type="file" accept="image/*" class="absolute inset-0 cursor-pointer opacity-0" onchange={onFileChange} />
						<div class="pointer-events-none flex flex-col items-center gap-3 text-center">
							<div class="rounded-full bg-elevated p-4 transition-transform group-hover:scale-110">
								<svg class="size-8 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5"
										d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/>
								</svg>
							</div>
							<p class="font-medium">Перетащите фото яйца</p>
							<p class="text-sm text-muted">или нажмите для выбора</p>
							<p class="text-xs text-muted/60">PNG · JPG · WEBP</p>
						</div>
					</label>
				{:else}
					<!-- Preview -->
					<div class="relative overflow-hidden rounded-xl border border-rim">
						<img src={previewUrl} alt="превью" class="h-64 w-full object-cover" />
						<div class="absolute inset-0 bg-linear-to-t from-black/60 to-transparent"></div>
						<div class="absolute bottom-0 left-0 right-0 flex items-center justify-between p-3">
							<span class="truncate text-sm font-medium">{file?.name}</span>
							<button
								onclick={reset}
								class="ml-2 shrink-0 rounded-lg bg-black/50 px-3 py-1.5 text-xs text-white backdrop-blur hover:bg-black/70"
							>Сменить</button>
						</div>
					</div>
				{/if}

				<!-- Кнопка -->
				<button
					onclick={analyze}
					disabled={!canAnalyze}
					class="rounded-xl px-6 py-3.5 font-semibold transition-all active:scale-[0.98]
						{canAnalyze
							? 'bg-amber-500 text-black hover:bg-amber-400 cursor-pointer'
							: 'bg-elevated text-muted cursor-not-allowed'}"
				>
					{#if analyzing}
						<span class="flex items-center justify-center gap-2">
							<svg class="size-4 animate-spin" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
							</svg>
							Анализируется...
						</span>
					{:else}
						🔬 Запустить анализ
					{/if}
				</button>

				<!-- Pipeline steps -->
				<div class="rounded-xl border border-rim bg-card p-4">
					<p class="mb-3 text-xs font-semibold uppercase tracking-widest text-muted">Pipeline</p>
					<div class="space-y-3">
						{#each [
							{ n: 1, icon: '📸', name: 'Moondream2',     desc: 'Визуальный анализ скорлупы' },
							{ n: 2, icon: '🧠', name: 'Qwen2.5-1.5B',  desc: 'Вердикт в формате JSON'    },
						] as step}
							<div class="flex items-center gap-3">
								<div class="flex size-7 shrink-0 items-center justify-center rounded-full border text-xs font-bold transition-colors
									{stage === step.n && analyzing ? 'border-amber-400 bg-amber-400/10 text-amber-400'
									: stage > step.n || (done && stage >= step.n) ? 'border-green-400 bg-green-400/10 text-green-400'
									: 'border-rim bg-elevated text-muted'}">
									{#if stage > step.n || (done && stage >= step.n && !analyzing)}
										✓
									{:else if stage === step.n && analyzing}
										<span class="animate-blink">●</span>
									{:else}
										{step.n}
									{/if}
								</div>
								<div>
									<p class="text-sm font-medium">{step.icon} {step.name}</p>
									<p class="text-xs text-muted">{step.desc}</p>
								</div>
							</div>
						{/each}
					</div>
				</div>
			</section>

			<!-- ── Правая колонка: лог + результат ──────── -->
			<section class="flex flex-col gap-4">
				<h2 class="text-xs font-semibold uppercase tracking-widest text-muted">📊 Прогресс</h2>

				<!-- Лог -->
				<div class="min-h-52 overflow-y-auto rounded-xl border border-rim bg-card p-4 font-mono text-xs">
					{#if logs.length === 0}
						<p class="text-muted">Ожидание запуска...</p>
					{:else}
						<div class="space-y-1.5">
							{#each logs as log}
								{#if log.kind === 'stage'}
									<div class="mt-3 flex items-center gap-2 first:mt-0">
										<span class="text-amber-400">▶</span>
										<span class="font-semibold text-amber-400">{log.text}</span>
									</div>
								{:else}
									<div class="flex items-start gap-2 pl-4">
										<span class="mt-px shrink-0 text-muted">→</span>
										<span class="text-muted">{log.text}</span>
									</div>
								{/if}
							{/each}
							{#if analyzing}
								<div class="pl-4"><span class="animate-blink text-amber-400">_</span></div>
							{/if}
						</div>
					{/if}
				</div>

				<!-- Результат -->
				{#if result}
					{@const isGood  = result.quality === 'good'}
					{@const isError = result.quality === 'error'}
					<div class="overflow-hidden rounded-xl border transition-colors
						{isGood ? 'border-green-500/40' : isError ? 'border-yellow-500/40' : 'border-red-500/40'}">

						<!-- Вердикт -->
						<div class="flex items-center justify-between px-5 py-4
							{isGood ? 'bg-green-500/10' : isError ? 'bg-yellow-500/10' : 'bg-red-500/10'}">
							<div class="flex items-center gap-3">
								<span class="text-4xl">{isGood ? '✅' : isError ? '⚠️' : '❌'}</span>
								<div>
									<p class="text-xs text-muted">Итоговый вердикт</p>
									<p class="text-2xl font-black tracking-tight
										{isGood ? 'text-green-400' : isError ? 'text-yellow-400' : 'text-red-400'}">
										{result.verdict_ru}
									</p>
								</div>
							</div>
							{#if result.confidence && CONFIDENCE[result.confidence]}
								<div class="text-right">
									<p class="text-xs text-muted">Уверенность</p>
									<p class="font-semibold {CONFIDENCE[result.confidence].cls}">
										{CONFIDENCE[result.confidence].label}
									</p>
								</div>
							{/if}
						</div>

						<div class="space-y-4 p-5">
							<!-- Дефекты -->
							{#if result.defects_found.length > 0}
								<div>
									<p class="mb-2 text-xs font-semibold uppercase tracking-widest text-muted">Дефекты</p>
									<div class="flex flex-wrap gap-2">
										{#each result.defects_found as defect}
											<span class="rounded-full border border-red-500/30 bg-red-500/10 px-3 py-1 text-xs text-red-400">
												{defect}
											</span>
										{/each}
									</div>
								</div>
							{:else if isGood}
								<span class="rounded-full border border-green-500/30 bg-green-500/10 px-3 py-1 text-xs text-green-400">
									Дефекты не обнаружены
								</span>
							{/if}

							<!-- Обоснование -->
							{#if result.reasoning}
								<div>
									<p class="mb-1 text-xs font-semibold uppercase tracking-widest text-muted">Обоснование</p>
									<p class="text-sm leading-relaxed text-muted">{result.reasoning}</p>
								</div>
							{/if}
						</div>
					</div>
				{/if}

				<!-- Ошибка -->
				{#if errorMsg}
					<div class="rounded-xl border border-red-500/30 bg-red-500/5 p-4">
						<p class="mb-1 text-xs font-semibold text-red-400">Ошибка</p>
						<p class="text-sm text-muted">{errorMsg}</p>
						<p class="mt-2 text-xs text-muted/60">Убедитесь, что Python-сервис запущен: <code>python service/main.py</code></p>
					</div>
				{/if}
			</section>
		</div>
	</main>

	<footer class="mt-8 border-t border-rim py-5">
		<p class="text-center text-xs text-muted">Moondream2 · Qwen2.5-1.5B · SvelteKit · FastAPI</p>
	</footer>
</div>

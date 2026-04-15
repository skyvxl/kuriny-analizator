import type { RequestHandler } from "./$types";
import { env } from "$env/dynamic/private";

const PYTHON_URL = env.PYTHON_SERVICE_URL ?? "http://localhost:8000";

export const POST: RequestHandler = async ({ request }) => {
    const formData = await request.formData();

    try {
        const upstream = await fetch(`${PYTHON_URL}/analyze`, {
            method: "POST",
            body: formData,
        });

        if (!upstream.ok || !upstream.body) {
            const msg = JSON.stringify({
                type: "error",
                message: `Сервис вернул ${upstream.status}`,
            });
            return new Response(`data: ${msg}\n\n`, {
                headers: { "Content-Type": "text/event-stream" },
            });
        }

        // Прозрачно прокидываем SSE-поток Python → браузер
        return new Response(upstream.body, {
            headers: {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                Connection: "keep-alive",
                "X-Accel-Buffering": "no",
            },
        });
    } catch (err) {
        const message =
            err instanceof Error ? err.message : "Python-сервис недоступен";
        const msg = JSON.stringify({ type: "error", message });
        return new Response(`data: ${msg}\n\n`, {
            headers: { "Content-Type": "text/event-stream" },
        });
    }
};

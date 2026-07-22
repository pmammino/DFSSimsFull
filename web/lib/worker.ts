// Server-side helper: the single place that knows the worker's URL.
// The browser never talks to the worker directly — it calls this app's
// /api/* route handlers, which forward here. That hides the worker origin and
// gives us one spot to add auth headers / API keys later.
export const WORKER_API_URL =
  process.env.WORKER_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

const WORKER_API_KEY = process.env.WORKER_API_KEY || "";

export async function workerFetch(
  path: string,
  init?: RequestInit,
): Promise<Response> {
  const headers = new Headers(init?.headers);
  if (WORKER_API_KEY) headers.set("x-api-key", WORKER_API_KEY);
  return fetch(`${WORKER_API_URL}${path}`, {
    ...init,
    headers,
    // Artifacts are refreshed out-of-band; let the worker own caching.
    cache: "no-store",
  });
}

// Proxy a worker JSON response straight through, preserving status.
export async function proxyJson(path: string): Promise<Response> {
  try {
    const upstream = await workerFetch(path);
    const body = await upstream.text();
    return new Response(body, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch {
    return new Response(
      JSON.stringify({
        detail:
          "Worker unreachable. Is the FastAPI service running and WORKER_API_URL set?",
      }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  }
}

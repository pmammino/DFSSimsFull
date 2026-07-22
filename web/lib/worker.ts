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

// Forward a request to the worker and return its response verbatim (status +
// JSON body preserved). Optionally pass method/body/contentType for writes.
export async function proxy(
  path: string,
  opts: { method?: string; body?: BodyInit; contentType?: string } = {},
): Promise<Response> {
  try {
    const headers: Record<string, string> = {};
    if (opts.contentType) headers["content-type"] = opts.contentType;
    const upstream = await workerFetch(path, {
      method: opts.method ?? "GET",
      body: opts.body,
      headers,
    });
    const text = await upstream.text();
    return new Response(text, {
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

// Back-compat alias for GET proxies.
export function proxyJson(path: string): Promise<Response> {
  return proxy(path);
}

// Proxy a non-JSON response (e.g. a CSV download), preserving the upstream
// content-type and content-disposition so the browser downloads it correctly.
export async function proxyRaw(path: string): Promise<Response> {
  try {
    const upstream = await workerFetch(path);
    const buf = await upstream.arrayBuffer();
    const headers = new Headers();
    const ct = upstream.headers.get("content-type");
    const cd = upstream.headers.get("content-disposition");
    if (ct) headers.set("content-type", ct);
    if (cd) headers.set("content-disposition", cd);
    return new Response(buf, { status: upstream.status, headers });
  } catch {
    return new Response("worker unreachable", { status: 502 });
  }
}

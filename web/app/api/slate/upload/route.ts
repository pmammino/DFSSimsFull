import { workerFetch } from "@/lib/worker";

export const dynamic = "force-dynamic";

// Forward the multipart upload straight through to the worker.
export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const upstream = await workerFetch("/slate/upload", {
      method: "POST",
      body: form,
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "content-type": "application/json" },
    });
  } catch {
    return new Response(
      JSON.stringify({ detail: "Worker unreachable during upload." }),
      { status: 502, headers: { "content-type": "application/json" } },
    );
  }
}

import { proxy } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const body = await req.text();
  return proxy(`/run/${encodeURIComponent(id)}/export`, {
    method: "POST",
    body,
    contentType: "application/json",
  });
}

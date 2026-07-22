import { proxy } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string; c: string }> },
) {
  const { id, c } = await params;
  return proxy(
    `/run/${encodeURIComponent(id)}/candidate/${encodeURIComponent(c)}/place-distribution`,
  );
}

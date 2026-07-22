import { proxy } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  return proxy(`/run/${encodeURIComponent(id)}/facets`);
}

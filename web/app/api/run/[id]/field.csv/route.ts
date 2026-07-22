import { proxyRaw } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  return proxyRaw(`/run/${encodeURIComponent(id)}/field.csv`);
}

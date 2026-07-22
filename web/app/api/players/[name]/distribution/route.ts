import { proxyJson } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function GET(
  req: Request,
  { params }: { params: Promise<{ name: string }> },
) {
  const { name } = await params;
  const { searchParams } = new URL(req.url);
  const nbins = searchParams.get("nbins") ?? "40";
  const q = new URLSearchParams({ nbins });
  return proxyJson(
    `/players/${encodeURIComponent(name)}/distribution?${q.toString()}`,
  );
}

import { proxyJson } from "@/lib/worker";

export const dynamic = "force-dynamic";

export function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const kind = searchParams.get("kind") ?? "all";
  const search = searchParams.get("search") ?? "";
  const q = new URLSearchParams({ kind, search });
  return proxyJson(`/players?${q.toString()}`);
}

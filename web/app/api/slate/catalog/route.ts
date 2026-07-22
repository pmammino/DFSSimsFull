import { proxy } from "@/lib/worker";

export const dynamic = "force-dynamic";

export function GET(req: Request) {
  const { searchParams } = new URL(req.url);
  const refresh = searchParams.get("refresh") === "true" ? "?refresh=true" : "";
  return proxy(`/slate/catalog${refresh}`);
}

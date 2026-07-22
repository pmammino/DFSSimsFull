import { proxyJson } from "@/lib/worker";

export const dynamic = "force-dynamic";

export function GET() {
  return proxyJson("/status");
}

import { proxy } from "@/lib/worker";

export const dynamic = "force-dynamic";

export function GET() {
  return proxy("/slate/sample");
}

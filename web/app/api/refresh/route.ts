import { proxy } from "@/lib/worker";

export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const body = await req.text();
  return proxy("/refresh", {
    method: "POST",
    body,
    contentType: "application/json",
  });
}

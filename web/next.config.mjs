/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // WORKER_API_URL is read at request time by the /api proxy route handlers,
  // so no build-time inlining is needed. See lib/worker.ts.
};

export default nextConfig;

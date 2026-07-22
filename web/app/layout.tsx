import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DFS Contest Simulator",
  description:
    "MLB DFS contest simulator — correlated DraftKings sims, field modeling, and portfolio EV.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-rw-ink text-white antialiased">
        <div className="mx-auto max-w-[1400px] px-4 py-4">{children}</div>
      </body>
    </html>
  );
}

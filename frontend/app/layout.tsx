import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "plan4me — collective knowledge from video",
  description:
    "Turn hours of interviews and talks into one evidence-backed knowledge report.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

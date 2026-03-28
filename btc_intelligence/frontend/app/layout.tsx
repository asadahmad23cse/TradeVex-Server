import "./globals.css";
import type { ReactNode } from "react";

import NavBar from "../components/NavBar";

export const metadata = {
  title: "Advanced Bitcoin Trade Intelligence",
  description: "Production-grade BTC signal intelligence"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <NavBar />
        <div className="pageWrap">{children}</div>
      </body>
    </html>
  );
}

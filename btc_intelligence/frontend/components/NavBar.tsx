"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Signal" },
  { href: "/analysis", label: "Analysis" },
  { href: "/history", label: "History" },
  { href: "/regime", label: "Regime" },
  { href: "/monitoring", label: "Monitoring" },
  { href: "/options", label: "Options" },
];

export default function NavBar() {
  const pathname = usePathname();
  return (
    <nav className="navBar">
      {links.map((l) => (
        <Link key={l.href} href={l.href} className={pathname === l.href ? "navLink active" : "navLink"}>
          {l.label}
        </Link>
      ))}
    </nav>
  );
}

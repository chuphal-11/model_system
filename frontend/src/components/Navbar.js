"use client";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import styles from "./Navbar.module.css";

export default function Navbar() {
  const pathname = usePathname();
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => {});
  }, []);

  const links = [
    { href: "/", label: "Dashboard" },
    { href: "/upload", label: "Upload" },
    { href: "/live", label: "Live Camera" },
  ];

  return (
    <nav className={styles.nav}>
      <div className={styles.inner}>
        <Link href="/" className={styles.logo}>
          <div className={styles.logoBox}>NN</div>
          <span className={styles.logoText}>NEURAL NEXUS</span>
        </Link>

        <div className={styles.links}>
          {links.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`${styles.link} ${pathname === link.href ? styles.active : ""}`}
            >
              {link.label}
            </Link>
          ))}
        </div>

        <div className={styles.badge}>
          <span
            className={styles.dot}
            style={{ background: status?.ready ? "#16a34a" : "#dc2626" }}
          />
          <span className={styles.badgeText}>
            {status?.gpu_name
              ? `${status.gpu_name} • ${status.models_loaded} models`
              : "Connecting…"}
          </span>
        </div>
      </div>
    </nav>
  );
}

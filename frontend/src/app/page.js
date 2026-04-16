"use client";
import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import styles from "./page.module.css";

export default function Dashboard() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  async function fetchStatus() {
    try {
      const r = await fetch("/api/status");
      setStatus(await r.json());
    } catch { }
  }

  return (
    <>
      <Navbar />
      <main className={styles.main}>
        <div className={styles.header}>
          <div>
            <span className={styles.tag}>System Monitor</span>
            <h1 className={styles.title}>DASHBOARD</h1>
            <p className={styles.subtitle}>
              Real-time overview of the Neural Nexus AI Classroom Intelligence pipeline.
            </p>
          </div>
          <button className={styles.refreshBtn} onClick={fetchStatus}>
            ↻ REFRESH
          </button>
        </div>

        <div className={styles.statsGrid}>
          <div className={styles.statCard}>
            <div className={`${styles.statIcon} ${styles.iconGreen}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <div className={styles.statInfo}>
              <span className={styles.statLabel}>STATUS</span>
              <span className={styles.statValue}>{status?.ready ? "Online" : "Offline"}</span>
            </div>
            <div className={styles.statBar} style={{ background: status?.ready ? "var(--success)" : "var(--danger)" }} />
          </div>

          <div className={styles.statCard}>
            <div className={`${styles.statIcon} ${styles.iconYellow}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
            </div>
            <div className={styles.statInfo}>
              <span className={styles.statLabel}>GPU</span>
              <span className={styles.statValue}>{status?.gpu_name || "N/A"}</span>
            </div>
          </div>

          <div className={styles.statCard}>
            <div className={`${styles.statIcon} ${styles.iconBlue}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>
            </div>
            <div className={styles.statInfo}>
              <span className={styles.statLabel}>MODELS LOADED</span>
              <span className={styles.statValue}>{status?.models_loaded ?? "—"}</span>
            </div>
          </div>

          <div className={styles.statCard}>
            <div className={`${styles.statIcon} ${styles.iconPurple}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
            </div>
            <div className={styles.statInfo}>
              <span className={styles.statLabel}>VRAM USED</span>
              <span className={styles.statValue}>
                {status?.gpu_memory_used_mb
                  ? `${status.gpu_memory_used_mb} / ${status.gpu_memory_total_mb} MB`
                  : "—"}
              </span>
            </div>
          </div>

          <div className={styles.statCard}>
            <div className={`${styles.statIcon} ${styles.iconDark}`}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>
            </div>
            <div className={styles.statInfo}>
              <span className={styles.statLabel}>DEVICE</span>
              <span className={styles.statValue}>{status?.device || "—"}</span>
            </div>
          </div>
        </div>

        <h2 className={styles.sectionTitle}>CAPABILITIES</h2>
        <div className={styles.featureGrid}>
          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M15 10l4.553-2.276A1 1 0 0 1 21 8.618v6.764a1 1 0 0 1-1.447.894L15 14M5 18h8a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2z"/></svg>
            </div>
            <h3 className={styles.featureName}>VIDEO ANALYSIS</h3>
            <p className={styles.featureDesc}>
              Upload classroom videos for comprehensive analysis. 9 AI models detect activities, track individuals, and extract behavioral patterns.
            </p>
          </div>

          <div className={`${styles.featureCard} ${styles.featureCardDark}`}>
            <div className={styles.featureIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            </div>
            <h3 className={styles.featureName}>LIVE MONITORING</h3>
            <p className={styles.featureDesc}>
              Stream live webcam feed through the full AI pipeline. Real-time annotated frames with engagement metrics delivered via WebSocket.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
            </div>
            <h3 className={styles.featureName}>PERSON TRACKING</h3>
            <p className={styles.featureDesc}>
              DeepSORT-powered multi-person tracking maintains unique IDs across frames using appearance-based re-identification.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
            </div>
            <h3 className={styles.featureName}>ENGAGEMENT METRICS</h3>
            <p className={styles.featureDesc}>
              Quantified engagement scores, participation rates, disruption indices, and teacher interaction ratios computed in real-time.
            </p>
          </div>

          <div className={styles.featureCard}>
            <div className={styles.featureIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
            </div>
            <h3 className={styles.featureName}>BEHAVIOR INFERENCE</h3>
            <p className={styles.featureDesc}>
              Classifies classroom state into modes: interactive, lecture, collaborative, distracted, or idle through temporal event analysis.
            </p>
          </div>

          <div className={`${styles.featureCard} ${styles.featureCardBeige}`}>
            <div className={styles.featureIcon}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
            </div>
            <h3 className={styles.featureName}>TEMPORAL SMOOTHING</h3>
            <p className={styles.featureDesc}>
              Sliding-window buffers stabilize noisy per-frame detections, confirming activities only when observed consistently across frames.
            </p>
          </div>
        </div>
      </main>
    </>
  );
}

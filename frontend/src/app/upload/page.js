"use client";
import { useState, useRef } from "react";
import Navbar from "@/components/Navbar";
import styles from "./upload.module.css";

export default function UploadPage() {
  const [dragover, setDragover] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [job, setJob] = useState(null);
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef();
  const pollRef = useRef();

  async function handleFile(file) {
    if (!file) return;
    setUploading(true);
    setJob(null);
    setJobId(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const r = await fetch("/api/upload", { method: "POST", body: form });
      const data = await r.json();
      if (data.error) { alert(data.error); setUploading(false); return; }

      setJobId(data.job_id);
      setJob({ status: "queued", filename: data.filename, progress: 0, total_frames: 0 });
      setUploading(false);

      // Poll for status
      pollRef.current = setInterval(async () => {
        try {
          const sr = await fetch(`/api/jobs/${data.job_id}`);
          const sdata = await sr.json();
          setJob(sdata);
          if (sdata.status === "done" || sdata.status === "error") {
            clearInterval(pollRef.current);
          }
        } catch {}
      }, 1000);
    } catch (e) {
      alert("Upload failed: " + e.message);
      setUploading(false);
    }
  }

  const pct = job?.total_frames > 0 ? Math.round((job.progress / job.total_frames) * 100) : 0;
  const agg = job?.aggregate;

  return (
    <>
      <Navbar />
      <main className={styles.main}>
        <div className={styles.header}>
          <span className={styles.tag}>Video Analysis</span>
          <h1 className={styles.title}>UPLOAD</h1>
          <p className={styles.subtitle}>
            Upload a classroom recording for comprehensive AI-powered behavioral analysis.
            The pipeline runs 9 detection models on your GPU.
          </p>
        </div>

        {/* Upload Zone */}
        {!job && (
          <div
            className={`${styles.uploadZone} ${dragover ? styles.dragover : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
            onDragLeave={() => setDragover(false)}
            onDrop={(e) => { e.preventDefault(); setDragover(false); handleFile(e.dataTransfer.files[0]); }}
            onClick={() => fileRef.current?.click()}
          >
            <input
              ref={fileRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv,.webm"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
            <div className={styles.uploadIcon}>
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <h3 className={styles.uploadTitle}>
              {uploading ? "UPLOADING…" : "DROP YOUR VIDEO HERE"}
            </h3>
            <p className={styles.uploadHint}>or click to browse • MP4, AVI, MOV, MKV, WebM</p>
          </div>
        )}

        {/* Processing Progress */}
        {job && job.status !== "done" && job.status !== "error" && (
          <div className={styles.processingCard}>
            <div className={styles.processHeader}>
              <h3>⚙️ PROCESSING {job.filename?.toUpperCase()}</h3>
              <span className={styles.processPercent}>{pct}%</span>
            </div>
            <div className={styles.progressBar}>
              <div className={styles.progressFill} style={{ width: `${pct}%` }} />
            </div>
            <p className={styles.processInfo}>
              Frame {job.progress} / {job.total_frames || "?"} • {job.status}
            </p>
          </div>
        )}

        {/* Error State */}
        {job?.status === "error" && (
          <div className={styles.errorCard}>
            <h3>❌ PROCESSING FAILED</h3>
            <p>{job.error}</p>
            <button className={styles.retryBtn} onClick={() => { setJob(null); setJobId(null); }}>
              TRY AGAIN
            </button>
          </div>
        )}

        {/* Results */}
        {job?.status === "done" && agg && (
          <div className={styles.results}>
            <div className={styles.resultsHeader}>
              <h2 className={styles.resultsTitle}>ANALYSIS COMPLETE</h2>
              <span className={styles.resultsTime}>
                Processed in {job.processing_time_seconds}s
              </span>
            </div>

            {/* Metric Cards */}
            <div className={styles.metricsGrid}>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>CLASSROOM STATE</span>
                <div className={styles.stateBadge}>
                  {agg.dominant_state?.toUpperCase().replace(/_/g, " ")}
                </div>
              </div>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>ENGAGEMENT</span>
                <span className={`${styles.metricValue} ${styles.green}`}>
                  {(agg.avg_engagement * 100).toFixed(1)}%
                </span>
                <div className={styles.metricBar}>
                  <div className={styles.metricBarFill} style={{ width: `${agg.avg_engagement * 100}%`, background: "var(--success)" }} />
                </div>
              </div>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>PARTICIPATION</span>
                <span className={`${styles.metricValue} ${styles.blue}`}>
                  {(agg.avg_participation * 100).toFixed(1)}%
                </span>
                <div className={styles.metricBar}>
                  <div className={styles.metricBarFill} style={{ width: `${agg.avg_participation * 100}%`, background: "var(--info)" }} />
                </div>
              </div>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>DISRUPTION</span>
                <span className={`${styles.metricValue} ${styles.red}`}>
                  {(agg.avg_disruption * 100).toFixed(1)}%
                </span>
                <div className={styles.metricBar}>
                  <div className={styles.metricBarFill} style={{ width: `${agg.avg_disruption * 100}%`, background: "var(--danger)" }} />
                </div>
              </div>
              <div className={styles.metricCard}>
                <span className={styles.metricLabel}>TEACHER INTERACTION</span>
                <span className={`${styles.metricValue} ${styles.yellow}`}>
                  {(agg.avg_teacher_interaction * 100).toFixed(1)}%
                </span>
                <div className={styles.metricBar}>
                  <div className={styles.metricBarFill} style={{ width: `${agg.avg_teacher_interaction * 100}%`, background: "var(--accent)" }} />
                </div>
              </div>
            </div>

            {/* State Distribution */}
            {agg.state_distribution && (
              <div className={styles.distCard}>
                <h3 className={styles.distTitle}>STATE DISTRIBUTION</h3>
                {Object.entries(agg.state_distribution)
                  .sort(([, a], [, b]) => b - a)
                  .map(([state, frac]) => (
                    <div key={state} className={styles.distRow}>
                      <span className={styles.distName}>{state}</span>
                      <div className={styles.distBar}>
                        <div className={styles.distFill} style={{ width: `${(frac * 100).toFixed(1)}%` }} />
                      </div>
                      <span className={styles.distPct}>{(frac * 100).toFixed(1)}%</span>
                    </div>
                  ))}
              </div>
            )}

            {/* Actions */}
            <div className={styles.actions}>
              {job.has_video && (
                <a href={`/api/jobs/${jobId}/video`} className={styles.downloadBtn} target="_blank">
                  ↓ DOWNLOAD ANNOTATED VIDEO
                </a>
              )}
              <a href={`/api/jobs/${jobId}/results`} className={styles.downloadBtnAlt} target="_blank">
                ↓ DOWNLOAD JSON RESULTS
              </a>
              <button className={styles.newBtn} onClick={() => { setJob(null); setJobId(null); }}>
                + NEW ANALYSIS
              </button>
            </div>
          </div>
        )}
      </main>
    </>
  );
}

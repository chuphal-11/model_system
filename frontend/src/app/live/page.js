"use client";
import { useState, useRef, useCallback } from "react";
import Navbar from "@/components/Navbar";
import styles from "./live.module.css";

export default function LivePage() {
  const [connected, setConnected] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [state, setState] = useState(null);
  const [events, setEvents] = useState([]);
  const [tracked, setTracked] = useState(0);
  const [inferenceMs, setInferenceMs] = useState(0);
  const wsRef = useRef(null);
  const imgRef = useRef(null);

  const startCamera = useCallback(() => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${protocol}://${location.host}/ws/camera`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) { alert("Camera error: " + data.error); stopCamera(); return; }
      if (data.type === "frame") {
        if (imgRef.current) imgRef.current.src = "data:image/jpeg;base64," + data.frame;
        setMetrics(data.metrics);
        setState({ name: data.classroom_state, confidence: data.state_confidence });
        setEvents(data.active_events || []);
        setTracked(data.tracked_entities);
        setInferenceMs(data.inference_ms);
      }
    };

    ws.onclose = () => stopCamera();
    ws.onerror = () => stopCamera();
  }, []);

  const stopCamera = useCallback(() => {
    if (wsRef.current) {
      try { wsRef.current.send("stop"); } catch {}
      try { wsRef.current.close(); } catch {}
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const pct = (v) => (v * 100).toFixed(1);

  return (
    <>
      <Navbar />
      <main className={styles.main}>
        <div className={styles.header}>
          <div>
            <span className={styles.tag}>Real-Time</span>
            <h1 className={styles.title}>LIVE CAMERA</h1>
            <p className={styles.subtitle}>
              Stream your webcam through the full AI pipeline with real-time annotations and metrics.
            </p>
          </div>
          <div className={styles.controls}>
            {!connected ? (
              <button className={styles.startBtn} onClick={startCamera}>
                ▶ START CAMERA
              </button>
            ) : (
              <button className={styles.stopBtn} onClick={stopCamera}>
                ■ STOP CAMERA
              </button>
            )}
          </div>
        </div>

        <div className={styles.layout}>
          {/* Video Feed */}
          <div className={styles.feedContainer}>
            <div className={styles.feed}>
              {!connected && (
                <div className={styles.placeholder}>
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
                    <path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/>
                  </svg>
                  <p>Click START CAMERA to begin live analysis</p>
                </div>
              )}
              <img
                ref={imgRef}
                className={styles.feedImg}
                style={{ display: connected ? "block" : "none" }}
                alt="Live Feed"
              />
              {connected && (
                <div className={styles.overlays}>
                  <span className={styles.liveBadge}>● LIVE</span>
                  <span className={styles.msBadge}>{inferenceMs}ms</span>
                </div>
              )}
            </div>
          </div>

          {/* Metrics Sidebar */}
          <div className={styles.sidebar}>
            {/* State */}
            <div className={`${styles.sideCard} ${styles.sideCardDark}`}>
              <span className={styles.sideLabel}>CLASSROOM STATE</span>
              <span className={styles.sideStateValue}>
                {state ? state.name.replace(/_/g, " ").toUpperCase() : "—"}
              </span>
              <span className={styles.sideStateSub}>
                {state ? `Confidence: ${pct(state.confidence)}%` : "Waiting for data"}
              </span>
            </div>

            {/* Engagement */}
            <div className={styles.sideCard}>
              <span className={styles.sideLabel}>ENGAGEMENT</span>
              <span className={`${styles.sideValue} ${styles.green}`}>
                {metrics ? `${pct(metrics.engagement_score)}%` : "—"}
              </span>
              <div className={styles.gauge}>
                <div className={styles.gaugeFill} style={{
                  width: metrics ? `${pct(metrics.engagement_score)}%` : "0%",
                  background: "var(--success)"
                }} />
              </div>
            </div>

            {/* Participation */}
            <div className={styles.sideCard}>
              <span className={styles.sideLabel}>PARTICIPATION</span>
              <span className={`${styles.sideValue} ${styles.blue}`}>
                {metrics ? `${pct(metrics.participation_rate)}%` : "—"}
              </span>
              <div className={styles.gauge}>
                <div className={styles.gaugeFill} style={{
                  width: metrics ? `${pct(metrics.participation_rate)}%` : "0%",
                  background: "var(--info)"
                }} />
              </div>
            </div>

            {/* Disruption */}
            <div className={styles.sideCard}>
              <span className={styles.sideLabel}>DISRUPTION</span>
              <span className={`${styles.sideValue} ${styles.red}`}>
                {metrics ? `${pct(metrics.disruption_index)}%` : "—"}
              </span>
              <div className={styles.gauge}>
                <div className={styles.gaugeFill} style={{
                  width: metrics ? `${pct(metrics.disruption_index)}%` : "0%",
                  background: "var(--danger)"
                }} />
              </div>
            </div>

            {/* Teacher Interaction */}
            <div className={styles.sideCard}>
              <span className={styles.sideLabel}>TEACHER INTERACTION</span>
              <span className={`${styles.sideValue} ${styles.yellow}`}>
                {metrics ? `${pct(metrics.teacher_interaction_ratio)}%` : "—"}
              </span>
              <div className={styles.gauge}>
                <div className={styles.gaugeFill} style={{
                  width: metrics ? `${pct(metrics.teacher_interaction_ratio)}%` : "0%",
                  background: "var(--accent)"
                }} />
              </div>
            </div>

            {/* Tracked & Events */}
            <div className={styles.sideCard}>
              <span className={styles.sideLabel}>TRACKED PERSONS</span>
              <span className={styles.sideValue} style={{ color: "var(--text-primary)" }}>{tracked}</span>
              <div className={styles.eventsBox}>
                {events.length > 0
                  ? events.map((e, i) => <span key={i} className={styles.eventTag}>{e}</span>)
                  : <span className={styles.noEvents}>No active events</span>
                }
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}

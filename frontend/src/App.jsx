import React, { useState, useRef, useEffect } from 'react';

const WS_RADIO_URL    = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/transcribe/radio/`;
const WS_PLAYBACK_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/transcribe/playback/`;
const RADIO_SAMPLE_RATE = 16000;

const SOURCE_LABELS = { audio_device: 'Audio Device', ip_stream: 'IP Stream', udp: 'UDP' };

export default function App() {
  const [running, setRunning]                 = useState(false);
  const [audioEnabled, setAudioEnabled]       = useState(false);
  const [radioStatus, setRadioStatus]         = useState('stopped');
  const [radioSource, setRadioSource]         = useState(null);
  const [radioTranscript, setRadioTranscript] = useState('');
  const [recording, setRecording]             = useState(false);
  const [recordStatus, setRecordStatus]       = useState(null);

  const radioWsRef    = useRef(null);
  const playbackWsRef = useRef(null);
  const radioBodyRef  = useRef(null);
  const runningRef    = useRef(false);
  const reconnectRef  = useRef(null);
  const audioCtxRef   = useRef(null);
  const nextStartRef  = useRef(0);

  useEffect(() => {
    if (radioBodyRef.current)
      radioBodyRef.current.scrollTop = radioBodyRef.current.scrollHeight;
  }, [radioTranscript]);

  function scheduleAudio(arrayBuffer) {
    const ctx = audioCtxRef.current;
    if (!ctx || ctx.state === 'closed') return;
    const int16   = new Int16Array(arrayBuffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;
    const buf = ctx.createBuffer(1, float32.length, RADIO_SAMPLE_RATE);
    buf.copyToChannel(float32, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const start = Math.max(ctx.currentTime, nextStartRef.current);
    src.start(start);
    nextStartRef.current = start + buf.duration;
  }

  function ensureAudioCtx() {
    if (!audioCtxRef.current || audioCtxRef.current.state === 'closed') {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RADIO_SAMPLE_RATE });
      audioCtxRef.current = ctx;
      nextStartRef.current = ctx.currentTime;
      setAudioEnabled(true);
    }
  }

  function handleAudioToggle() {
    if (!audioEnabled) {
      ensureAudioCtx();
    } else {
      audioCtxRef.current?.close();
      audioCtxRef.current = null;
      nextStartRef.current = 0;
      setAudioEnabled(false);
    }
  }

  function connect() {
    if (!runningRef.current) return;
    setRadioStatus('connecting');
    setRadioSource(null);

    const ws = new WebSocket(WS_RADIO_URL);
    ws.binaryType = 'arraybuffer';
    radioWsRef.current = ws;

    ws.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) { scheduleAudio(e.data); return; }
      const data = JSON.parse(e.data);
      if (data.error)                    { setRadioStatus('error'); return; }
      if (data.status === 'connected')   { setRadioStatus('listening'); setRadioSource(data.source); }
      if (data.text)                     setRadioTranscript(prev => prev + (prev ? '\n' : '') + data.text);
    };

    ws.onclose = () => {
      if (!runningRef.current) return;
      setRadioStatus('reconnecting');
      reconnectRef.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => setRadioStatus('error');
  }

  function handleStart() {
    runningRef.current = true;
    setRunning(true);
    connect();
  }

  async function handleRecord() {
    // Close any in-progress playback
    if (playbackWsRef.current) {
      playbackWsRef.current.onclose = null;
      playbackWsRef.current.close();
      playbackWsRef.current = null;
    }

    setRecording(true);
    setRecordStatus('processing');

    // Ensure audio context exists (Record click is a valid user gesture)
    ensureAudioCtx();

    // Open playback stream and transcription POST in parallel
    const playbackWs = new WebSocket(WS_PLAYBACK_URL);
    playbackWs.binaryType = 'arraybuffer';
    playbackWsRef.current = playbackWs;

    playbackWs.onmessage = (e) => {
      if (e.data instanceof ArrayBuffer) { scheduleAudio(e.data); return; }
      // text frames are status/error — nothing to display
    };
    playbackWs.onclose = () => { playbackWsRef.current = null; };

    try {
      const resp = await fetch('/api/transcribe/record/', { method: 'POST' });
      const data = await resp.json();
      if (data.error) {
        setRecordStatus('error: ' + data.error);
      } else if (data.transcription) {
        setRadioTranscript(prev => prev + (prev ? '\n' : '') + '[File] ' + data.transcription);
        setRecordStatus('done');
      } else {
        setRecordStatus('no speech detected');
      }
    } catch {
      setRecordStatus('failed');
    } finally {
      setRecording(false);
    }
  }

  function handleStop() {
    runningRef.current = false;
    setRunning(false);
    setRadioStatus('stopped');
    setRadioSource(null);
    clearTimeout(reconnectRef.current);
    if (radioWsRef.current) { radioWsRef.current.onclose = null; radioWsRef.current.close(); radioWsRef.current = null; }
  }

  return (
    <div className="shell">
      <section className="panel">
        <div className="panel-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <span className="panel-label">RADIO TRANSCRIPT</span>
            <RadioStatusPill status={radioStatus} source={radioSource} />
          </div>
          <div className="panel-actions">
            {!running
              ? <button className="start-btn" onClick={handleStart}>▶ Start Live</button>
              : <button className="stop-btn"  onClick={handleStop}>⏹ Stop</button>
            }
            <button className="start-btn" onClick={handleRecord} disabled={recording}>
              {recording ? '⏳ Processing…' : '⏺ Record'}
            </button>
            {recordStatus && (
              <span style={{ fontSize: '0.75rem', color: recordStatus === 'done' ? 'var(--clr-active)' : '#e74c3c' }}>
                {recordStatus}
              </span>
            )}
            <button
              className={audioEnabled ? 'audio-btn audio-btn--on' : 'audio-btn'}
              onClick={handleAudioToggle}
            >
              {audioEnabled ? '🔇 Disable Audio' : '🔊 Enable Audio'}
            </button>
            {radioTranscript && (
              <button className="clear-btn" onClick={() => setRadioTranscript('')}>Clear</button>
            )}
          </div>
        </div>
        <div className="panel-body" ref={radioBodyRef}>
          {radioTranscript
            ? <p className="transcript-text" style={{ whiteSpace: 'pre-wrap' }}>{radioTranscript}</p>
            : <p className="transcript-empty">
                {radioStatus === 'listening'  ? 'Monitoring radio — transmissions will appear here…'
                : radioStatus === 'error'     ? 'Radio source unavailable.'
                : radioStatus === 'stopped'   ? 'Press Start to begin transcription.'
                : 'Connecting to radio source…'}
              </p>
          }
        </div>
      </section>
    </div>
  );
}

function RadioStatusPill({ status, source }) {
  const sourceLabel = source ? ` · ${SOURCE_LABELS[source] ?? source}` : '';
  const configs = {
    connecting:   { label: 'Connecting…',            cls: 'transcribing' },
    listening:    { label: `Listening${sourceLabel}`, cls: 'recording'    },
    reconnecting: { label: 'Reconnecting…',           cls: 'idle'         },
    error:        { label: 'No source configured',    cls: 'idle'         },
    stopped:      { label: 'Stopped',                 cls: 'idle'         },
  };
  const { label, cls } = configs[status] ?? configs.stopped;
  return (
    <div className={`status status--${cls}`} style={{ fontSize: '0.8rem' }}>
      <span className="status-dot" />
      Radio: {label}
    </div>
  );
}

import React, { useState, useRef, useEffect } from 'react';

const WS_RADIO_URL = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/transcribe/radio/`;
const RADIO_SAMPLE_RATE = 16000;

const SOURCE_LABELS = { audio_device: 'Audio Device', ip_stream: 'IP Stream', udp: 'UDP' };

export default function App() {
  const [running, setRunning]                 = useState(false);
  const [audioEnabled, setAudioEnabled]       = useState(false);
  const [radioStatus, setRadioStatus]         = useState('stopped');
  const [radioSource, setRadioSource]         = useState(null);
  const [radioTranscript, setRadioTranscript] = useState('');

  const radioWsRef    = useRef(null);
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

  function handleAudioToggle() {
    if (!audioEnabled) {
      const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RADIO_SAMPLE_RATE });
      audioCtxRef.current = ctx;
      nextStartRef.current = ctx.currentTime;
      setAudioEnabled(true);
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
      if (data.error)           { setRadioStatus('error'); return; }
      if (data.status === 'connected') { setRadioStatus('listening'); setRadioSource(data.source); }
      if (data.text)            setRadioTranscript(prev => prev + (prev ? '\n' : '') + data.text);
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
              ? <button className="start-btn" onClick={handleStart}>▶ Start</button>
              : <button className="stop-btn"  onClick={handleStop}>⏹ Stop</button>
            }
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

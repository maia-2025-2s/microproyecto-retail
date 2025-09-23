import React, { useMemo, useState, useEffect, type JSX } from "react";

type Row = { date: string; store: string; item: string; sales: number };
type PredRow = { date: string; store: string; item: string; yhat: number };
type HistoryPoint = { date: string; sales: number };
type ForecastPoint = { date: string; yhat: number };

const fmtDate = (d: Date): string => d.toISOString().slice(0, 10);
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);
const stdev = (xs: number[]) => {
  if (!xs.length) return 0;
  const m = mean(xs);
  const v = mean(xs.map((x) => (x - m) ** 2));
  return Math.sqrt(v);
};

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (inQuotes && line[i + 1] === '"') {
        cur += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (ch === "," && !inQuotes) {
      out.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

function parseCsv(text: string): Row[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];
  const header = splitCsvLine(lines[0]).map((h) => h.toLowerCase());
  const idxDate = header.indexOf("date");
  const idxStore = header.indexOf("store");
  const idxItem = header.indexOf("item");
  const idxSales = header.indexOf("sales");
  if ([idxDate, idxStore, idxItem, idxSales].some((i) => i === -1)) return [];
  const rows: Row[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCsvLine(lines[i]);
    if (!cols.length) continue;
    const date = cols[idxDate];
    const store = String(cols[idxStore]);
    const item = String(cols[idxItem]);
    const sales = Number(cols[idxSales]);
    if (!date || Number.isNaN(sales)) continue;
    rows.push({ date, store, item, sales });
  }
  rows.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : 0));
  return rows;
}

function parsePredCsv(text: string): PredRow[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) return [];
  const header = splitCsvLine(lines[0]).map((h) => h.toLowerCase());
  const idxDate = header.indexOf("date");
  const idxStore = header.indexOf("store");
  const idxItem = header.indexOf("item");
  const idxYhat = header.indexOf("yhat");
  if ([idxDate, idxStore, idxItem, idxYhat].some((i) => i === -1)) return [];
  const out: PredRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = splitCsvLine(lines[i]);
    if (!cols.length) continue;
    const date = cols[idxDate];
    const store = String(cols[idxStore]);
    const item = String(cols[idxItem]);
    const yhat = Number(cols[idxYhat]);
    if (!date || Number.isNaN(yhat)) continue;
    out.push({ date, store, item, yhat });
  }
  return out;
}

const genHistory = (days = 150): HistoryPoint[] => {
  const out: HistoryPoint[] = [];
  const today = new Date();
  for (let i = days; i >= 1; i--) {
    const d = new Date(today);
    d.setDate(today.getDate() - i);
    const dow = d.getDay();
    const base = 15 + (dow === 0 || dow === 6 ? 6 : 0);
    const trend = (days - i) * 0.02;
    const noise = (Math.random() - 0.5) * 2.5;
    out.push({ date: fmtDate(d), sales: Math.max(0, Math.round(base + trend + noise)) });
  }
  return out;
};

export default function SimpleDemandMvp(): JSX.Element {
  const [rows, setRows] = useState<Row[]>([]);
  const [preds, setPreds] = useState<PredRow[]>([]);
  const [predSource, setPredSource] = useState<string>("");

  const [storeId, setStoreId] = useState<string>("");
  const [itemId, setItemId] = useState<string>("");

  // opciones dinámicas desde API
  const [storeOptions, setStoreOptions] = useState<string[]>([]);
  const [itemOptions, setItemOptions] = useState<string[]>([]);

  // controles extra
  const [horizon, setHorizon] = useState<number>(14);
  const [stock, setStock] = useState<number>(80);
  const [leadTime, setLeadTime] = useState<number>(5);

  // cargar opciones desde API
  useEffect(() => {
    fetch("http://localhost:8000/options")
      .then((res) => res.json())
      .then((data) => {
        setStoreOptions(data.stores.map(String));
        setItemOptions(data.items.map(String));
        if (!storeId && data.stores.length > 0) setStoreId(String(data.stores[0]));
        if (!itemId && data.items.length > 0) setItemId(String(data.items[0]));
      })
      .catch((err) => console.error("Error cargando /options:", err));
  }, []);

  // cargar predicciones dummy desde CSV
  useEffect(() => {
    fetch("/predictions.csv")
      .then((r) => (r.ok ? r.text() : Promise.reject()))
      .then((t) => {
        const p = parsePredCsv(t);
        if (p.length) {
          setPreds(p);
          setPredSource("public/predictions.csv");
        }
      })
      .catch(() => {});
  }, []);

  // fetch a la API real
  useEffect(() => {
    if (storeId && itemId) {
      const url = `http://localhost:8000/predict?store=${storeId}&item=${itemId}`;
      fetch(url)
        .then((res) => (res.ok ? res.json() : Promise.reject("Fallo en la respuesta de la API")))
        .then((data) => {
          if (!Array.isArray(data)) return;
          const parsed = data
            .filter((r) => r.date && !Number.isNaN(Number(r.yhat)))
            .map((r) => ({
              date: r.date,
              store: String(storeId),
              item: String(itemId),
              yhat: Number(r.yhat),
            }));
          if (parsed.length) {
            setPreds(parsed);
            setPredSource("API local");
          }
        })
        .catch((err) => console.error("Error al consultar API:", err));
    }
  }, [storeId, itemId]);

  // cálculos
  const syntheticHistory = useMemo(() => genHistory(150), []);
  const history: HistoryPoint[] = useMemo(() => {
    if (rows.length && storeId && itemId) {
      const f = rows.filter((r) => r.store === storeId && r.item === itemId);
      if (f.length) return f.map((r) => ({ date: r.date, sales: r.sales }));
    }
    return syntheticHistory;
  }, [rows, storeId, itemId, syntheticHistory]);

  const predictedFc: ForecastPoint[] = useMemo(() => {
    if (!preds.length || !storeId || !itemId) return [];
    return preds
      .filter((p) => p.store === storeId && p.item === itemId)
      .map((p) => ({ date: p.date, yhat: p.yhat }));
  }, [preds, storeId, itemId]);

  const base = useMemo(() => Math.round(mean(history.slice(-7).map((x) => x.sales)) || 10), [history]);
  const fallbackFc: ForecastPoint[] = useMemo(() => {
    const start = new Date();
    return Array.from({ length: 28 }, (_, i) => {
      const d = new Date(start);
      d.setDate(start.getDate() + (i + 1));
      return { date: fmtDate(d), yhat: base };
    });
  }, [base]);

  const forecast: ForecastPoint[] = useMemo(() => {
    const f = predictedFc.length ? predictedFc : fallbackFc;
    return f.slice(0, 28);
  }, [predictedFc, fallbackFc]);

  const horizonFc = useMemo(() => forecast.slice(0, horizon), [forecast, horizon]);

  const demandNext7 = useMemo(() => horizonFc.slice(0, 7).reduce((a, b) => a + b.yhat, 0), [horizonFc]);
  const demandLead = useMemo(
    () => horizonFc.slice(0, Math.min(leadTime, horizonFc.length)).reduce((a, b) => a + b.yhat, 0),
    [horizonFc, leadTime]
  );
  const safety = useMemo(
    () => Math.round(0.5 * stdev(history.slice(-28).map((x) => x.sales)) * Math.sqrt(Math.max(1, leadTime))),
    [history, leadTime]
  );
  const suggestedOrder = Math.max(0, demandLead + safety - stock);
  const risk = useMemo(() => (stock - demandLead < 0 ? "Alto" : "Bajo"), [stock, demandLead]);

  const tableRows = useMemo(() => {
    let sim = stock;
    return horizonFc.map((d, idx) => {
      sim -= d.yhat;
      return { idx: idx + 1, ...d, sim, alert: sim < 0 ? "Riesgo" : "OK" } as const;
    });
  }, [horizonFc, stock]);

  const sparkData = useMemo(() => {
    const hist = history.slice(-30).map((h) => h.sales);
    const fc = horizonFc.map((f) => f.yhat);
    return [...hist, ...fc];
  }, [history, horizonFc]);

  return (
    <div style={{ padding: 24, fontFamily: "Inter, system-ui, sans-serif", lineHeight: 1.35 }}>
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 8 }}>Predicción de Demanda — RETAIL</h1>
      <p style={{ color: "#555", marginBottom: 16 }}>
        Flujo: predicciones desde <code>public/predictions.csv</code> o API.
      </p>

      {/* Store / Item */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, maxWidth: 1200, marginBottom: 12 }}>
        <label style={{ display: "grid", gap: 6 }}>
          <span>Store</span>
          <select value={storeId} onChange={(e) => setStoreId(e.target.value)} style={sel}>
            {storeOptions.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Item</span>
          <select value={itemId} onChange={(e) => setItemId(e.target.value)} style={sel}>
            {itemOptions.map((it) => (
              <option key={it} value={it}>{it}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Horizonte / Stock / Lead Time */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, maxWidth: 720 }}>
        <label style={{ display: "grid", gap: 6 }}>
          <span>Horizonte (días)</span>
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} style={sel}>
            <option value={7}>7</option>
            <option value={14}>14</option>
            <option value={28}>28</option>
          </select>
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Stock actual</span>
          <input type="number" value={stock} onChange={(e) => setStock(Number(e.target.value))} style={inp} />
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Lead time (días)</span>
          <input type="number" value={leadTime} onChange={(e) => setLeadTime(Number(e.target.value))} style={inp} />
        </label>
      </div>

      {/* KPIs */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, marginTop: 16, maxWidth: 960 }}>
        <Kpi title="Demanda próxima semana" value={demandNext7} />
        <Kpi title="Riesgo de quiebre" value={risk} />
        <Kpi title="Pedido sugerido HOY" value={suggestedOrder} />
      </div>

      {/* Sparkline */}
      <div style={{ ...card, marginTop: 16, maxWidth: 960 }}>
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Histórico (30d) + Forecast</div>
        <Sparkline data={sparkData} />
      </div>

      {/* Tabla */}
      <div style={{ ...card, marginTop: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Pronóstico</div>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr>
              <th>#</th><th>Fecha</th><th>Demanda</th><th>Stock sim.</th><th>Alerta</th>
            </tr>
          </thead>
          <tbody>
            {tableRows.map((r) => (
              <tr key={r.date}>
                <td>{r.idx}</td>
                <td>{r.date}</td>
                <td>{r.yhat}</td>
                <td>{r.sim}</td>
                <td>{r.alert}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// estilos
const card: React.CSSProperties = { border: "1px solid #eee", borderRadius: 12, padding: 16, background: "#fff" };
const inp: React.CSSProperties = { border: "1px solid #ddd", borderRadius: 8, padding: 8 };
const sel: React.CSSProperties = { ...inp };

function Kpi({ title, value }: { title: string; value: string | number }) {
  return (
    <div style={card}>
      <div style={{ fontSize: 12, color: "#777", marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 20, fontWeight: 600 }}>{String(value)}</div>
    </div>
  );
}

function Sparkline({ data, width = 420, height = 80 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return <svg width={width} height={height} />;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const norm = (v: number) => (max === min ? height / 2 : height - ((v - min) / (max - min)) * height);
  const step = width / (data.length - 1);
  const d = data.map((v, i) => `${i === 0 ? "M" : "L"}${i * step},${norm(v)}`).join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block", border: "1px solid #eee", borderRadius: 8 }}>
      <path d={d} fill="none" stroke="#555" strokeWidth={2} />
    </svg>
  );
}



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

  const [horizon, setHorizon] = useState<number>(14);
  const [stock, setStock] = useState<number>(80);
  const [leadTime, setLeadTime] = useState<number>(5);

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

  const hasRows = rows.length > 0;
  const hasPreds = preds.length > 0;

  const storeOptions = useMemo(() => {
    const src = hasRows ? rows.map((r) => r.store) : hasPreds ? preds.map((p) => p.store) : [];
    return Array.from(new Set(src)).sort((a, b) => String(a).localeCompare(String(b)));
  }, [rows, preds, hasRows, hasPreds]);

  const itemOptions = useMemo(() => {
    const src = hasRows
      ? rows.filter((r) => !storeId || r.store === storeId).map((r) => r.item)
      : hasPreds
      ? preds.filter((p) => !storeId || p.store === storeId).map((p) => p.item)
      : [];
    return Array.from(new Set(src)).sort((a, b) => String(a).localeCompare(String(b)));
  }, [rows, preds, storeId, hasRows, hasPreds]);

  useEffect(() => {
    if ((rows.length || preds.length) && !storeId && storeOptions[0]) setStoreId(storeOptions[0]);
  }, [rows, preds, storeOptions, storeId]);

  useEffect(() => {
    if ((rows.length || preds.length) && !itemId && itemOptions[0]) setItemId(itemOptions[0]);
  }, [rows, preds, itemOptions, itemId]);

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

  const Sparkline = ({ data, width = 420, height = 80 }: { data: number[]; width?: number; height?: number }) => {
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
  };

  const downloadCSV = () => {
    const header = "date,forecast,simStock,alert\n";
    const rowsCsv = tableRows.map((r) => `${r.date},${r.yhat},${r.sim},${r.alert}`).join("\n");
    const blob = new Blob([header + rowsCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const tag = storeId && itemId ? `_${storeId}_${itemId}` : "";
    a.download = `forecast_${horizon}d${tag}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const onPickCsv: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      const first = text.split(/\r?\n/)[0] || "";
      const header = splitCsvLine(first).map((h) => h.toLowerCase());
      const hasSales = header.includes("sales");
      const hasYhat = header.includes("yhat");
      if (hasSales) {
        const parsed = parseCsv(text);
        if (!parsed.length) {
          alert("CSV inválido. Se esperan columnas: date,store,item,sales");
          return;
        }
        setRows(parsed);
      } else if (hasYhat) {
        const parsedP = parsePredCsv(text);
        if (!parsedP.length) {
          alert("CSV inválido de predicciones. Se esperan columnas: date,store,item,yhat");
          return;
        }
        setPreds(parsedP);
        setPredSource(f.name);
        alert("Se detectó un CSV de predicciones; en producción usamos public/predictions.csv. Se cargará para poder probar.");
      } else {
        alert("CSV inválido. Se esperan columnas: date,store,item,sales (histórico) o date,store,item,yhat (predicciones)");
      }
    };
    reader.readAsText(f);
  };

  const usingDataset = rows.length > 0 && !!storeId && !!itemId;
  const usingPreds = preds.length > 0;

  const [testMsg, setTestMsg] = useState<string>("");
  useEffect(() => {
    try {
      const results: string[] = [];
      const p1 = parseCsv("date,store,item,sales\n2025-01-01,1,1,10\n2025-01-02,1,1,12");
      results.push(p1.length === 2 && p1[0].sales === 10 && p1[1].date === "2025-01-02" ? "✓ LF split" : "✗ LF split");
      const p2 = parseCsv("date,store,item,sales\r\n2025-01-01,2,5,7\r\n2025-01-02,2,5,9");
      results.push(p2.length === 2 && p2[0].store === "2" && p2[1].sales === 9 ? "✓ CRLF split" : "✗ CRLF split");
      const p3 = parseCsv("Date,Store,Item,Sales\n2025-03-01,3,8,4");
      results.push(p3.length === 1 && p3[0].item === "8" ? "✓ Headers case-insensitive" : "✗ Headers case-insensitive");
      const p4 = parseCsv('date,store,item,sales\n2025-02-01,"A, Norte",SKU-1,11');
      results.push(p4.length === 1 && p4[0].store === "A, Norte" ? "✓ Quotes & comma" : "✗ Quotes & comma");
      const pp = parsePredCsv("date,store,item,yhat\n2025-01-01,1,1,12");
      results.push(pp.length === 1 && pp[0].yhat === 12 ? "✓ Pred schema" : "✗ Pred schema");
      setTestMsg(results.join("  •  "));
    } catch (e) {
      setTestMsg("Tests error: " + (e as Error).message);
    }
  }, []);

  return (
    <div style={{ padding: 24, fontFamily: "Inter, system-ui, sans-serif", lineHeight: 1.35 }}>
      <h1 style={{ fontSize: 22, fontWeight: 600, marginBottom: 8 }}>Predicción de Demanda — RETAIL</h1>
      <p style={{ color: "#555", marginBottom: 16 }}>Flujo: predicciones desde <code>public/predictions.csv</code> + histórico opcional.</p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 12, maxWidth: 1200, marginBottom: 12 }}>
        <label style={{ display: "grid", gap: 6 }}>
          <span>
            Datos (CSV: <code>date,store,item,sales</code>)
          </span>
          <input type="file" accept=".csv" onChange={onPickCsv} />
          <span style={{ color: "#777", fontSize: 11 }}>
            Si no se sube CSV, se usa histórico sintético. Las <b>predicciones</b> se leen de <code>public/predictions.csv</code>.
          </span>
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Store</span>
          <select value={storeId} onChange={(e) => setStoreId(e.target.value)} disabled={!rows.length && !preds.length} style={sel}>
            {storeOptions.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "grid", gap: 6 }}>
          <span>Item</span>
          <select value={itemId} onChange={(e) => setItemId(e.target.value)} disabled={!rows.length && !preds.length} style={sel}>
            {itemOptions.map((it) => (
              <option key={it} value={it}>
                {it}
              </option>
            ))}
          </select>
        </label>
      </div>

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

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 12, marginTop: 16, maxWidth: 960 }}>
        <Kpi title="Demanda próxima semana" value={demandNext7} />
        <Kpi title="Riesgo de quiebre" value={risk} />
        <Kpi title="Pedido sugerido HOY" value={suggestedOrder} />
        <button onClick={downloadCSV} style={{ ...card, cursor: "pointer" }}>
          <div style={{ fontSize: 12, color: "#777", marginBottom: 6 }}>Exportar</div>
          <div style={{ fontSize: 20, fontWeight: 600 }}>CSV</div>
        </button>
      </div>

      <div style={{ marginTop: 10, fontSize: 12 }}>
        <span style={{ color: usingDataset ? "#0a7" : "#777" }}>
          {usingDataset ? `Usando dataset cargado (store=${storeId}, item=${itemId})` : "Usando datos sintéticos (sin CSV)"}
        </span>
        {"  •  "}
        <span style={{ color: usingPreds ? "#0a7" : "#777" }}>
          {usingPreds ? `Predicciones activas (${predSource || "public/predictions.csv"})` : "Sin predicciones externas (baseline)"}
        </span>
      </div>

      <div style={{ ...card, marginTop: 16, maxWidth: 960 }}>
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Histórico (30d) + Forecast</div>
        <Sparkline data={sparkData} />
      </div>

      <div style={{ ...card, marginTop: 16 }}>
        <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 8 }}>Pronóstico (próximos {horizon} días)</div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
            <thead>
              <tr style={{ textAlign: "left" }}>
                <th style={th}>#</th>
                <th style={th}>Fecha</th>
                <th style={th}>Demanda</th>
                <th style={th}>Stock sim.</th>
                <th style={th}>Alerta</th>
              </tr>
            </thead>
            <tbody>
              {tableRows.map((r) => (
                <tr key={r.date}>
                  <td style={td}>{r.idx}</td>
                  <td style={td}>{r.date}</td>
                  <td style={td}>{r.yhat}</td>
                  <td style={td}>{r.sim}</td>
                  <td style={td}>{r.alert}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginTop: 10, fontSize: 12, color: testMsg.startsWith("✓") ? "#0a7" : "#555" }}>Tests CSV: {testMsg}</div>
    </div>
  );
}

const card: React.CSSProperties = { border: "1px solid #eee", borderRadius: 12, padding: 16, background: "#fff" };
const th: React.CSSProperties = { padding: "8px 6px", borderBottom: "1px solid #eee" };
const td: React.CSSProperties = { padding: "8px 6px", borderTop: "1px solid #f5f5f5" };
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

/* Bot Futures — Dashboard JS */

const API = "";  // mesmo host
let ws = null;
let currentSymbol = "BTC/USDC:USDC";
let currentStrategy = "wyckoff";
let currentMode = "testnet";
let chart = null;

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------

function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => log("Conectado ao servidor", "ok");

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    handleEvent(msg);
  };

  ws.onclose = () => {
    log("Conexão perdida — reconectando em 5s...", "warn");
    setTimeout(connectWS, 5000);
  };

  // Ping a cada 30s para manter conexão viva
  setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, 30000);
}

function handleEvent(msg) {
  switch (msg.type) {
    case "connected":
      updateBotStatus(msg.running);
      updateConfig(msg.config);
      updateStats(msg.stats);
      break;
    case "bot_started":
      updateBotStatus(true);
      log(`Bot iniciado — estratégia: ${msg.strategy}`, "ok");
      break;
    case "bot_stopped":
      updateBotStatus(false);
      log("Bot parado", "warn");
      break;
    case "mode_changed":
      currentMode = msg.mode;
      renderModeToggle();
      log(`Modo alterado: ${msg.mode.toUpperCase()}`, "warn");
      break;
    case "leverage_changed":
      document.getElementById("leverage-val").textContent = msg.leverage + "x";
      document.getElementById("leverage-slider").value = msg.leverage;
      break;
    case "strategy_changed":
      currentStrategy = msg.strategy;
      document.getElementById("strategy-select").value = msg.strategy;
      break;
    case "signal_update":
      if (msg.symbol === currentSymbol) {
        renderConditions(msg);
      }
      break;
    case "trade_opened":
      log(`✅ Trade aberto: ${msg.symbol} ${msg.side} @ ${msg.entry} (${msg.leverage}x)`, "ok");
      loadTrades();
      break;
    case "position_update":
      log(msg.message || `Posição atualizada: ${msg.symbol}`, "info");
      loadStatus();
      break;
  }
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

async function api(path, method = "GET", body = null) {
  const opts = { method, headers: { "Content-Type": "application/json" } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

async function loadStatus() {
  try {
    const data = await api("/api/status");
    updateBotStatus(data.running);
    updateConfig(data.config);
    updateStats(data.stats);
    renderPositions(data.positions);
  } catch (e) {
    log(`Erro ao carregar status: ${e.message}`, "error");
  }
}

async function loadTrades() {
  try {
    const data = await api("/api/trades?limit=20");
    renderTrades(data.trades);
  } catch (e) {
    log(`Erro ao carregar trades: ${e.message}`, "error");
  }
}

async function loadCandles() {
  try {
    const tf = currentStrategy === "smc_swing" ? "4h" : "15m";
    const sym = encodeURIComponent(currentSymbol);
    const data = await api(`/api/candles?symbol=${sym}&timeframe=${tf}&limit=200`);
    renderChart(data.candles);
  } catch (e) {
    log(`Erro ao carregar candles: ${e.message}`, "error");
  }
}

async function scanSymbol() {
  log(`Analisando ${currentSymbol}...`, "info");
  try {
    const sym = currentSymbol.replace("/", "-");
    const data = await api(`/api/scan/${sym}?strategy=${currentStrategy}`);
    renderConditions({ ...data, symbol: currentSymbol, strategy: currentStrategy });
    log(`Análise concluída: ${data.signal} (R:R ${data.rr})`, data.signal !== "NONE" ? "ok" : "info");
  } catch (e) {
    log(`Erro ao analisar: ${e.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function updateBotStatus(running) {
  const btn = document.getElementById("bot-btn");
  const badge = document.getElementById("bot-status-badge");

  if (running) {
    btn.textContent = "⏹ Parar Bot";
    btn.className = "btn btn-stop";
    btn.onclick = () => botCommand("stop");
    badge.textContent = "RODANDO";
    badge.className = "badge badge-running";
  } else {
    btn.textContent = "▶ Iniciar Bot";
    btn.className = "btn btn-start";
    btn.onclick = () => botCommand("start");
    badge.textContent = "PARADO";
    badge.className = "badge badge-stopped";
  }
}

function updateConfig(cfg) {
  if (!cfg) return;
  currentMode = cfg.mode || "testnet";
  currentStrategy = cfg.strategy || "wyckoff";

  document.getElementById("strategy-select").value = currentStrategy;
  document.getElementById("leverage-slider").value = cfg.leverage || 3;
  document.getElementById("leverage-val").textContent = (cfg.leverage || 3) + "x";
  renderModeToggle();
}

function updateStats(stats) {
  if (!stats) return;
  const pnl = stats.total_pnl || 0;
  const el = document.getElementById("total-pnl");
  el.textContent = (pnl >= 0 ? "+" : "") + pnl.toFixed(2) + " USDC";
  el.className = "stat-value " + (pnl >= 0 ? "pos" : "neg");

  document.getElementById("win-rate").textContent =
    stats.total_trades > 0 ? stats.win_rate + "%" : "—";
  document.getElementById("total-trades").textContent = stats.total_trades || 0;
}

function renderModeToggle() {
  const testBtn = document.getElementById("btn-testnet");
  const liveBtn = document.getElementById("btn-live");
  testBtn.className = currentMode === "testnet" ? "active-testnet" : "";
  liveBtn.className = currentMode === "live" ? "active-live" : "";

  const badge = document.getElementById("exchange-mode-badge");
  badge.textContent = currentMode.toUpperCase();
  badge.className = "badge badge-" + currentMode;
}

function renderConditions(data) {
  const container = document.getElementById("conditions-list");
  const signalEl = document.getElementById("signal-badge");
  const signal = data.signal || "NONE";

  signalEl.textContent = signal;
  signalEl.className = "signal-badge signal-" + signal.toLowerCase();

  if (!data.conditions || data.conditions.length === 0) {
    container.innerHTML = '<div class="no-position">Aguardando análise...</div>';
    return;
  }

  container.innerHTML = data.conditions.map(c => `
    <div class="condition-item">
      <span class="cond-icon">${c.passed ? "✅" : "❌"}</span>
      <div class="cond-text">
        <div class="cond-name">${c.name}</div>
        <div class="cond-detail">${c.detail || ""}</div>
      </div>
    </div>
  `).join("");
}

function renderPositions(positions) {
  const el = document.getElementById("positions-container");
  if (!positions || positions.length === 0) {
    el.innerHTML = '<div class="no-position">Nenhuma posição aberta</div>';
    return;
  }

  el.innerHTML = positions.map(p => {
    const pnlClass = p.pnl_pct >= 0 ? "pos" : "neg";
    const pnlSign = p.pnl_pct >= 0 ? "+" : "";
    const progress = Math.min(100, Math.max(0, p.progress_pct));
    const sideColor = p.side === "long" ? "var(--green)" : "var(--red)";

    return `
      <div style="margin-bottom:16px; padding-bottom:16px; border-bottom:1px solid var(--border)">
        <div style="display:flex; justify-content:space-between; margin-bottom:8px">
          <span style="font-weight:700">${p.symbol}</span>
          <span style="color:${sideColor}; font-weight:700; text-transform:uppercase">${p.side} ${p.leverage}x</span>
        </div>
        <div class="position-grid">
          <div class="pos-item">
            <div class="pos-label">Entrada</div>
            <div class="pos-value">${p.entry?.toFixed(4)}</div>
          </div>
          <div class="pos-item">
            <div class="pos-label">Stop</div>
            <div class="pos-value" style="color:var(--red)">${p.stop_loss?.toFixed(4)}</div>
          </div>
          <div class="pos-item">
            <div class="pos-label">Target</div>
            <div class="pos-value" style="color:var(--green)">${p.take_profit?.toFixed(4)}</div>
          </div>
          <div class="pos-item">
            <div class="pos-label">P&L</div>
            <div class="pos-value ${pnlClass}">${pnlSign}${p.pnl_pct?.toFixed(1)}%</div>
          </div>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width:${progress}%; background:${progress >= 100 ? 'var(--green)' : 'var(--blue)'}"></div>
        </div>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px; display:flex; gap:12px">
          <span>${progress.toFixed(0)}% do alvo</span>
          ${p.breakeven_hit ? '<span style="color:var(--green)">🔒 Breakeven ativo</span>' : ""}
          ${p.partial_closed ? '<span style="color:var(--blue)">📊 Parcial fechado</span>' : ""}
        </div>
      </div>
    `;
  }).join("");
}

function renderTrades(trades) {
  const tbody = document.getElementById("trades-tbody");
  if (!trades || trades.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:16px">Sem trades registrados</td></tr>';
    return;
  }

  tbody.innerHTML = trades.map(t => {
    const pnl = t.pnl_usdt;
    const pnlClass = pnl == null ? "trade-open" : pnl >= 0 ? "trade-win" : "trade-loss";
    const pnlText = pnl == null ? "—" : (pnl >= 0 ? "+" : "") + pnl.toFixed(2);
    const date = new Date(t.timestamp).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" });
    const sideColor = t.side === "LONG" ? "var(--green)" : "var(--red)";

    return `<tr>
      <td>${date}</td>
      <td style="font-weight:700">${t.symbol?.replace("/USDC:USDC", "")}</td>
      <td style="color:${sideColor}">${t.side}</td>
      <td>${t.leverage}x</td>
      <td>${t.rr?.toFixed(1)}</td>
      <td class="${pnlClass}">${pnlText}</td>
    </tr>`;
  }).join("");
}

function renderChart(candles) {
  if (!candles || candles.length === 0) return;

  const timestamps = candles.map(c => c.t);
  const ohlc = {
    type: "candlestick",
    x: timestamps,
    open: candles.map(c => c.o),
    high: candles.map(c => c.h),
    low: candles.map(c => c.l),
    close: candles.map(c => c.c),
    name: currentSymbol.replace(":USDC", ""),
    increasing: { line: { color: "var(--green)" } },
    decreasing: { line: { color: "var(--red)" } },
  };

  const ema20 = {
    type: "scatter", mode: "lines", x: timestamps,
    y: candles.map(c => c.ema20),
    name: "EMA20", line: { color: "#f0c040", width: 1 }, opacity: 0.8,
  };
  const ema50 = {
    type: "scatter", mode: "lines", x: timestamps,
    y: candles.map(c => c.ema50),
    name: "EMA50", line: { color: "#7c6fcd", width: 1 }, opacity: 0.8,
  };
  const ema200 = {
    type: "scatter", mode: "lines", x: timestamps,
    y: candles.map(c => c.ema200),
    name: "EMA200", line: { color: "#58a6ff", width: 1.5 }, opacity: 0.9,
  };
  const bbUpper = {
    type: "scatter", mode: "lines", x: timestamps,
    y: candles.map(c => c.bb_upper),
    name: "BB Upper", line: { color: "#8b949e", width: 1, dash: "dot" }, opacity: 0.5,
  };
  const bbLower = {
    type: "scatter", mode: "lines", x: timestamps,
    y: candles.map(c => c.bb_lower),
    name: "BB Lower", line: { color: "#8b949e", width: 1, dash: "dot" }, opacity: 0.5,
    fill: "tonexty", fillcolor: "rgba(139,148,158,0.05)",
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#8b949e", size: 11 },
    xaxis: {
      gridcolor: "#21262d",
      showgrid: true,
      rangeslider: { visible: false },
      color: "#8b949e",
    },
    yaxis: {
      gridcolor: "#21262d",
      showgrid: true,
      color: "#8b949e",
      side: "right",
    },
    legend: { bgcolor: "transparent", font: { size: 10 } },
    margin: { l: 8, r: 60, t: 8, b: 30 },
    showlegend: true,
  };

  const config = { responsive: true, displayModeBar: false };

  if (chart) {
    Plotly.react("chart", [ohlc, bbUpper, bbLower, ema20, ema50, ema200], layout, config);
  } else {
    chart = true;
    Plotly.newPlot("chart", [ohlc, bbUpper, bbLower, ema20, ema50, ema200], layout, config);
  }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async function botCommand(action) {
  try {
    await api("/api/bot", "POST", { action, strategy: currentStrategy });
  } catch (e) {
    log(`Erro: ${e.message}`, "error");
  }
}

function toggleSection(sectionId, chevronId) {
  const section = document.getElementById(sectionId);
  const chevron = document.getElementById(chevronId);
  if (!section) return;
  const collapsed = section.classList.toggle("section-collapsed");
  if (chevron) chevron.classList.toggle("collapsed", collapsed);
}

async function switchMode(mode) {
  if (mode === "live") {
    const ok = confirm(
      "⚠️ ATENÇÃO: Você está prestes a ativar o modo LIVE.\n\n" +
      "O bot usará DINHEIRO REAL na Binance.\n\n" +
      "Confirma a mudança para LIVE?"
    );
    if (!ok) return;
  }
  try {
    await api("/api/config", "POST", { mode });
    currentMode = mode;
    renderModeToggle();
    log(`Modo alterado para ${mode.toUpperCase()}`, mode === "live" ? "warn" : "ok");
  } catch (e) {
    log(`Erro ao mudar modo: ${e.message}`, "error");
  }
}

async function setLeverage(val) {
  document.getElementById("leverage-val").textContent = val + "x";
  try {
    await api("/api/config", "POST", { leverage: parseInt(val) });
  } catch (e) {
    log(`Erro ao definir alavancagem: ${e.message}`, "error");
  }
}

async function setStrategy(val) {
  currentStrategy = val;
  try {
    await api("/api/config", "POST", { strategy: val });
    await loadCandles();
  } catch (e) {
    log(`Erro ao mudar estratégia: ${e.message}`, "error");
  }
}

function selectSymbol(symbol) {
  currentSymbol = symbol;
  document.querySelectorAll(".symbol-pill").forEach(el => {
    el.classList.toggle("active", el.dataset.symbol === symbol);
  });
  loadCandles();
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

function log(msg, level = "info") {
  const el = document.getElementById("log-output");
  const time = new Date().toLocaleTimeString("pt-BR");
  const line = document.createElement("div");
  line.className = `log-line ${level}`;
  line.textContent = `[${time}] ${msg}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;

  // Limita a 100 linhas
  while (el.children.length > 100) el.removeChild(el.firstChild);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  connectWS();
  await Promise.all([loadStatus(), loadTrades(), loadCandles()]);

  // Auto-refresh a cada 60s
  setInterval(() => {
    loadStatus();
    loadTrades();
  }, 60000);
});

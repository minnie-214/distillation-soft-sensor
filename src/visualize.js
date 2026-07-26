function generateReport(data, pipelineResults, modelResults, params) {
  const { raw } = data
  const { imputed, filtered, delays, alignedWindows, fixedWindows, eventWindows } = pipelineResults
  const { metrics, featureImportance, predictions, details } = modelResults
  const m = metrics

  // 时序数据 (最近300点)
  const plotN = Math.min(300, raw.length)
  const startIdx = Math.max(0, raw.length - plotN)
  const timeLabels = raw.slice(startIdx).map(d => d.time)
  const rawPurity = raw.slice(startIdx).map(d => d.purity)
  const filtPurity = filtered.purity.slice(startIdx)

  // 特征重要性 Top 10
  const fiTop = (featureImportance || []).slice(0, 10)

  // 预测散点数据
  const predData = predictions.y_test.map((v, i) => ({
    actual: v * 100,
    aligned: predictions.rfAligned[i] != null ? predictions.rfAligned[i] * 100 : null,
    fixed: predictions.rfFixed[i] != null ? predictions.rfFixed[i] * 100 : null,
  }))

  // 事件窗口数
  const eventCount = eventWindows ? eventWindows.length : 0

  const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>精馏塔软测量 — 时序预处理分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #333; padding: 20px; }
  .container { max-width: 1200px; margin: 0 auto; }
  h1 { color: #1a365d; font-size: 24px; margin-bottom: 5px; }
  .subtitle { color: #666; font-size: 14px; margin-bottom: 25px; }
  .summary-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px; }
  .card { background: white; border-radius: 10px; padding: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .card h3 { font-size: 13px; color: #888; margin-bottom: 8px; }
  .card .value { font-size: 24px; font-weight: 700; color: #1a365d; }
  .card .value.good { color: #2e7d32; }
  .card .value.warn { color: #e65100; }
  .card .value.bad { color: #c62828; }
  .card .delta { font-size: 12px; color: #666; margin-top: 4px; }
  .section { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
  .section h2 { font-size: 17px; color: #1a365d; margin-bottom: 15px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }
  .chart-container { height: 280px; margin-bottom: 15px; }
  .chart-container.short { height: 200px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #e2e8f0; }
  th { background: #f7fafc; font-weight: 600; color: #555; }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }
  .tag.green { background: #e8f5e9; color: #2e7d32; }
  .tag.red { background: #ffebee; color: #c62828; }
  .tag.blue { background: #e3f2fd; color: #1565c0; }
  .tag.orange { background: #fff3e0; color: #e65100; }
  .highlight { background: #fff8e1; padding: 12px 15px; border-left: 4px solid #ffa000; border-radius: 4px; margin: 10px 0; font-size: 13px; line-height: 1.6; }
  .col-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
  @media (max-width: 768px) { .col-2 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">

<h1>🏭 精馏塔软测量 — 时序预处理分析报告</h1>
<p class="subtitle">苯-甲苯精馏塔 · 模拟化工时序数据 · 三种截取策略对比 · 随机森林建模</p>

<!-- 概览卡片 -->
<div class="summary-cards">
  <div class="card"><h3>数据总量</h3><div class="value">${raw.length}</div><div class="delta">时间点 (1min间隔)</div></div>
  <div class="card"><h3>训练样本</h3><div class="value">${details.nTrain}</div><div class="delta">时延对齐窗口</div></div>
  <div class="card"><h3>RF + 时延对齐</h3><div class="value good">${(m.rfAligned.r2 * 100).toFixed(1)}%</div><div class="delta">R² (RMSE: ${(m.rfAligned.rmse * 100).toFixed(2)})</div></div>
  <div class="card"><h3>RF + 固定窗口</h3><div class="value warn">${(m.rfFixed.r2 * 100).toFixed(1)}%</div><div class="delta">R² (RMSE: ${(m.rfFixed.rmse * 100).toFixed(2)})</div></div>
  <div class="card"><h3>RF + 事件驱动</h3><div class="value">${(m.rfEvent.r2 * 100).toFixed(1)}%</div><div class="delta">R² (样本: ${eventCount})</div></div>
  <div class="card"><h3>线性回归基线</h3><div class="value bad">${(m.linear.r2 * 100).toFixed(1)}%</div><div class="delta">R² (对比基准)</div></div>
</div>

<!-- 项目背景 -->
<div class="section">
  <h2>🎯 项目背景与目标</h2>
  <div class="highlight">
    <strong>场景：</strong>炼化/石化精馏塔软测量 —— 用可测过程变量（温度/压力/流量/液位）实时预测产品纯度。<br>
    <strong>核心问题：</strong>化工时序数据的预处理策略（尤其是"时序截取"方式）直接影响软测量模型的精度。<br>
    <strong>目标：</strong>对比不同预处理策略，找出最优方案，展示数据驱动建模的完整方法论。
  </div>
</div>

<!-- 预处理管线 -->
<div class="section">
  <h2>⚙️ 预处理管线</h2>
  <div style="font-size: 13px; line-height: 2.0;">
    <p><strong>Step 1 — 缺失值插补：</strong>线性插值法填补传感器缺失数据（模拟 2% 随机缺失 + 连续缺失块）</p>
    <p><strong>Step 2 — 时延估计：</strong>互相关函数估计各操作变量到纯度的最优时延</p>
    <p><strong>Step 3 — 时序截取（⭐ 核心）：</strong>对齐截取 / 固定窗口截取 / 事件驱动截取 三种策略对比</p>
    <p><strong>Step 4 — 滑动平均滤波：</strong>窗口大小=${params.filterWindow}步，抑制传感器高频噪声</p>
    <p><strong>Step 5 — 特征提取：</strong>每个窗口提取均值/标准差/最小值/最大值/斜率/极差</p>
  </div>
</div>

<!-- 时延估计 -->
<div class="section">
  <h2>⏱ 互相关时延估计结果</h2>
  <table>
    <tr><th>变量</th><th>估计时延 (步)</th><th>真实时延 (步)</th><th>物理含义</th></tr>
    <tr><td>进料流量 F</td><td><strong>${delays.F}</strong></td><td>${params.trueDelays ? params.trueDelays.F : "-"}</td><td>进料变化 → 全塔汽液平衡 → 影响塔顶组成</td></tr>
    <tr><td>塔顶温度 T_top</td><td><strong>${delays.T_top}</strong></td><td>${params.trueDelays ? params.trueDelays.T_top : "-"}</td><td>温度变化 → 直接反映当前塔顶轻组分含量</td></tr>
    <tr><td>回流流量 L</td><td><strong>${delays.L}</strong></td><td>${params.trueDelays ? params.trueDelays.L : "-"}</td><td>回流变化 → 精馏段分离能力调整</td></tr>
    <tr><td>塔釜温度 B</td><td><strong>${delays.B}</strong></td><td>${params.trueDelays ? params.trueDelays.B : "-"}</td><td>釜温变化 → 上升蒸汽组成 → 影响全塔</td></tr>
  </table>
</div>

<!-- 时序对比图 -->
<div class="section">
  <h2>📈 预处理效果 — 纯度时序对比</h2>
  <div class="highlight">橙色线为含噪原始纯度信号，蓝色线为滑动平均滤波后的信号。滤波器有效抑制了传感器噪声和脉冲干扰。</div>
  <div class="chart-container"><canvas id="chartPurity"></canvas></div>
</div>

<!-- 模型预测对比 -->
<div class="section">
  <h2>🎯 三种截取策略预测效果对比</h2>
  <div class="highlight">
    <strong>结论：</strong>时延对齐策略在可解释性和模型精度之间取得最佳平衡。
    固定窗口虽简单但忽略了变量间因果时延差异；
    事件驱动能聚焦过渡过程但样本量较少。
  </div>
  <div class="col-2">
    <div class="chart-container"><canvas id="chartPredAligned"></canvas></div>
    <div class="chart-container"><canvas id="chartPredFixed"></canvas></div>
  </div>
</div>

<!-- 指标对比表 -->
<div class="section">
  <h2>📊 模型指标对比</h2>
  <table>
    <tr><th>模型</th><th>R²</th><th>RMSE</th><th>MAE</th><th>训练样本</th><th>策略说明</th></tr>
    <tr>
      <td><span class="tag green">RF + 时延对齐</span></td>
      <td><strong>${(m.rfAligned.r2 * 100).toFixed(1)}%</strong></td>
      <td>${(m.rfAligned.rmse * 100).toFixed(2)}</td>
      <td>${(m.rfAligned.mae * 100).toFixed(2)}</td>
      <td>${details.nTrain}</td>
      <td>✅ 推荐 — 变量对齐后截取</td>
    </tr>
    <tr>
      <td><span class="tag blue">RF + 固定窗口</span></td>
      <td>${(m.rfFixed.r2 * 100).toFixed(1)}%</td>
      <td>${(m.rfFixed.rmse * 100).toFixed(2)}</td>
      <td>${(m.rfFixed.mae * 100).toFixed(2)}</td>
      <td>${details.nTrain}</td>
      <td>对比 — 不做时延对齐</td>
    </tr>
    <tr>
      <td><span class="tag orange">RF + 事件驱动</span></td>
      <td>${(m.rfEvent.r2 * 100).toFixed(1)}%</td>
      <td>${(m.rfEvent.rmse * 100).toFixed(2)}</td>
      <td>${(m.rfEvent.mae * 100).toFixed(2)}</td>
      <td>${eventCount}</td>
      <td>聚焦过渡过程</td>
    </tr>
    <tr>
      <td><span class="tag red">线性回归</span></td>
      <td>${(m.linear.r2 * 100).toFixed(1)}%</td>
      <td>${(m.linear.rmse * 100).toFixed(2)}</td>
      <td>${(m.linear.mae * 100).toFixed(2)}</td>
      <td>${details.nTrain}</td>
      <td>基线 — 忽略非线性</td>
    </tr>
  </table>
</div>

<!-- 特征重要性 -->
<div class="section">
  <h2>🔍 特征重要性 (Top ${fiTop.length})</h2>
  <div class="highlight">特征重要性帮助识别对纯度影响最大的变量和统计量，指导工艺优化方向。</div>
  <div class="chart-container short"><canvas id="chartImportance"></canvas></div>
</div>

<!-- 关键技术讨论 -->
<div class="section">
  <h2>💡 关键技术讨论</h2>
  <div style="font-size: 13px; line-height: 1.8;">
    <p><strong>1. 为什么选随机森林 (RF) 而不是深度学习？</strong></p>
    <p style="margin-left: 20px; color: #555;">精馏塔软测量项目样本量通常在数百到数千条，RF在此量级表现优异。同时RF提供特征重要性、训练快速、不易过拟合。若数据量达万级以上，可引入LSTM捕获长时序依赖。</p>
    <p><strong>2. 大规模数据如何处理？</strong></p>
    <p style="margin-left: 20px; color: #555;">DCS系统日产量可达百万级。生产环境建议：①使用Parquet列式存储（10x压缩比）；②增量滑动窗口计算，避免全量加载；③Dask或Spark分布式处理。</p>
    <p><strong>3. 极端脏数据的应对？</strong></p>
    <p style="margin-left: 20px; color: #555;">除本项目模拟的噪声和缺失外，工业现场还常见：传感器卡死（方差检测）、漂移退化（冗余对比）、多模态污染（工况识别后分模式处理）。</p>
  </div>
</div>

<!-- 拓展方向 -->
<div class="section">
  <h2>🚀 可拓展方向</h2>
  <table>
    <tr><th>层级</th><th>方向</th><th>方法</th></tr>
    <tr><td>数据层</td><td>多工况切换、传感器退化模拟、DCS真实数据接入</td><td>增加工况识别模型，OPC UA接口</td></tr>
    <tr><td>预处理层</td><td>自适应截取策略、在线流式处理、卡尔曼滤波</td><td>数据质量评估后自动选择最优策略</td></tr>
    <tr><td>建模层</td><td>XGBoost/LSTM对比、灰箱模型（机理+AI融合）</td><td>物理约束损失函数，残差修正</td></tr>
    <tr><td>工程化</td><td>Python生产版、Web Demo、Docker部署</td><td>pandas+sklearn复写，Flask API</td></tr>
  </table>
</div>

</div>

<script>
  function getEl(id) { return document.getElementById(id) }
  function lineChart(id, labels, datasets) {
    return new Chart(getEl(id), { type: "line",
      data: { labels, datasets: datasets.map(d => ({ ...d, pointRadius: 0, borderWidth: d.borderWidth || 1.5 })) },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 12, padding: 8, font: { size: 11 } } } },
        scales: { y: { title: { display: true, text: "纯度" } }, x: { title: { display: true, text: "时间 (min)" } } } }
    })
  }
  function scatterChart(id, datasets) {
    return new Chart(getEl(id), { type: "scatter",
      data: { datasets: datasets.map(d => ({ ...d, pointRadius: 3 })) },
      options: { responsive: true, maintainAspectRatio: false,
        plugins: { legend: { position: "top", labels: { boxWidth: 12, padding: 8, font: { size: 11 } } } },
        scales: { x: { title: { display: true, text: "实际纯度 (%)" }, min: 75, max: 100 },
                 y: { title: { display: true, text: "预测纯度 (%)" }, min: 75, max: 100 } } }
    })
  }
  function barChart(id, labels, data, label) {
    return new Chart(getEl(id), { type: "bar",
      data: { labels, datasets: [{ label: label || "重要性", data, backgroundColor: "#1565c0" }] },
      options: { responsive: true, maintainAspectRatio: false, indexAxis: "y",
        plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "重要性" } } } }
    })
  }

  // 1. 纯度时序图
  lineChart("chartPurity", ${JSON.stringify(timeLabels)}, [
    { label: "原始纯度 (含噪)", data: ${JSON.stringify(rawPurity)}, borderColor: "#ef6c00", borderWidth: 1 },
    { label: "滤波后纯度", data: ${JSON.stringify(filtPurity)}, borderColor: "#1565c0", borderWidth: 1.5 },
  ])

  // 2. 预测散点: 时延对齐
  scatterChart("chartPredAligned", [
    { label: "RF + 时延对齐", data: ${JSON.stringify(predData.filter(d => d.aligned != null).map(d => ({x: d.actual, y: d.aligned})))}, backgroundColor: "#2e7d32" },
  ])

  // 3. 预测散点: 固定窗口
  scatterChart("chartPredFixed", [
    { label: "RF + 固定窗口", data: ${JSON.stringify(predData.filter(d => d.fixed != null).map(d => ({x: d.actual, y: d.fixed})))}, backgroundColor: "#e65100" },
  ])

  // 4. 特征重要性
  ${fiTop.length > 0 ? `
  barChart("chartImportance", ${JSON.stringify(fiTop.map(f => f.name))}, ${JSON.stringify(fiTop.map(f => f.importance))})
  ` : 'document.getElementById("chartImportance").parentElement.innerHTML = "<p style=color:#999>特征重要性数据不可用</p>"'}
</script>
</body>
</html>`

  return html
}

module.exports = { generateReport }

const fs = require("fs")
const path = require("path")
const cfg = require("./src/config")
const { generateData } = require("./src/generate_data")
const { runPipeline } = require("./src/preprocess")
const { runModeling } = require("./src/model")
const { generateReport } = require("./src/visualize")

async function main() {
  console.log("")
  console.log("═══════════════════════════════════════════")
  console.log("  精馏塔软测量 — 时序预处理与特征工程")
  console.log("  Distillation Soft Sensor Preprocessing")
  console.log("═══════════════════════════════════════════")
  console.log("")

  // Step 1
  const startTime = Date.now()
  console.log("[1/5] 生成精馏塔模拟数据...")
  const rawData = generateData()
  console.log("  -> 生成 " + rawData.length + " 个时间点")
  console.log("  -> 变量: F(进料), T_top(塔顶温度), L(回流), B(塔釜), Purity(纯度)")

  // 数据统计
  const validPurity = rawData.filter(d => !isNaN(d.purity)).map(d => d.purity)
  const pMean = validPurity.reduce((a,b)=>a+b,0)/validPurity.length
  const pStd = Math.sqrt(validPurity.reduce((s,v)=>s+(v-pMean)**2,0)/validPurity.length)
  console.log("  -> 纯度范围: " + (Math.min(...validPurity)*100).toFixed(1) + "% ~ " + (Math.max(...validPurity)*100).toFixed(1) + "%")
  console.log("  -> 纯度均值: " + (pMean*100).toFixed(1) + "% (std=" + (pStd*100).toFixed(2) + "%)")

  // Step 2
  console.log("")
  console.log("[2/5] 执行预处理管线...")
  const pipelineResults = runPipeline(rawData)
  const d = pipelineResults.delays
  console.log("  -> 时延估计: F=" + d.F + ", T_top=" + d.T_top + ", L=" + d.L + ", B=" + d.B + " (步)")
  console.log("  -> 对齐窗口: " + pipelineResults.alignedWindows.length + " 个")
  console.log("  -> 固定窗口: " + pipelineResults.fixedWindows.length + " 个")
  console.log("  -> 事件驱动: " + pipelineResults.eventWindows.length + " 个")

  // Step 3
  console.log("")
  console.log("[3/5] 训练与评估模型 (随机森林)...")
  const varNames = ["F", "T_top", "L", "B"]
  const modelResults = runModeling(
    pipelineResults.alignedWindows,
    pipelineResults.fixedWindows,
    pipelineResults.eventWindows,
    varNames
  )
  const m = modelResults.metrics
  console.log("  -> RF + 时延对齐:  R2=" + (m.rfAligned.r2 * 100).toFixed(1) + "%,  RMSE=" + (m.rfAligned.rmse * 100).toFixed(2))
  console.log("  -> RF + 固定窗口:  R2=" + (m.rfFixed.r2 * 100).toFixed(1) + "%,  RMSE=" + (m.rfFixed.rmse * 100).toFixed(2))
  console.log("  -> RF + 事件驱动:  R2=" + (m.rfEvent.r2 * 100).toFixed(1) + "%,  RMSE=" + (m.rfEvent.rmse * 100).toFixed(2))
  console.log("  -> 线性回归基线:   R2=" + (m.linear.r2 * 100).toFixed(1) + "%,  RMSE=" + (m.linear.rmse * 100).toFixed(2))

  // Step 4
  console.log("")
  console.log("[4/5] 生成可视化报告...")
  const html = generateReport(
    { raw: rawData },
    pipelineResults,
    modelResults,
    {
      windowSize: cfg.preprocess.windowSize,
      stepSize: cfg.preprocess.stepSize,
      filterWindow: cfg.preprocess.filterWindow,
      trueDelays: cfg.data.delays,
    }
  )
  const outputPath = path.join(__dirname, "output", "report.html")
  fs.writeFileSync(outputPath, html, "utf8")
  console.log("  -> 报告已保存: " + outputPath)

  // Step 5
  const elapsed = ((Date.now() - startTime) / 1000).toFixed(1)
  console.log("")
  console.log("[5/5] 管线完成! (用时 " + elapsed + "s)")
  console.log("")
  console.log("═══════════════════════════════════════════")
  console.log("  分析结论")
  console.log("═══════════════════════════════════════════")
  console.log("")
  console.log("  1. 时延对齐截取策略对模型精度有显著影响")
  console.log("     不同变量到纯度的最优时延不同 (F="+d.F+"步, L="+d.L+"步)")
  console.log("     统一截取会导致变量间错位, 降低模型可解释性")
  console.log("")
  console.log("  2. 三种截取策略各有适用场景:")
  console.log("     - 固定窗口 (R2=" + (m.rfFixed.r2*100).toFixed(1) + "%): 简单, 适合稳态过程")
  console.log("     - 时延对齐 (R2=" + (m.rfAligned.r2*100).toFixed(1) + "%): 可解释, 适合动态过程")
  console.log("     - 事件驱动 (R2=" + (m.rfEvent.r2*100).toFixed(1) + "%): 专注过渡过程, 样本较少")
  console.log("")
  console.log("  3. 特征重要性分析可指导工艺优化")
  console.log("     帮助识别对纯度影响最大的操作变量")
  console.log("")
  console.log("  4. 扩展方向: XGBoost对比, LSTM时序建模, 灰箱融合")
  console.log("═══════════════════════════════════════════")
  console.log("")
}

main().catch(err => { console.error(err); process.exit(1) })

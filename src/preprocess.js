const cfg = require("./config")

/**
 * 线性插值填充缺失值
 */
function imputeLinear(data, key) {
  const result = new Array(data.length)
  let lastValid = null, lastIdx = null
  for (let i = 0; i < data.length; i++) {
    const val = data[i][key]
    if (isNaN(val)) {
      let nextValid = null, nextIdx = null
      for (let j = i + 1; j < data.length; j++) {
        if (!isNaN(data[j][key])) { nextValid = data[j][key]; nextIdx = j; break }
      }
      if (lastValid !== null && nextValid !== null) {
        const ratio = (i - lastIdx) / (nextIdx - lastIdx)
        result[i] = lastValid + (nextValid - lastValid) * ratio
      } else if (lastValid !== null) result[i] = lastValid
      else if (nextValid !== null) result[i] = nextValid
      else result[i] = 0
    } else {
      result[i] = val; lastValid = val; lastIdx = i
    }
  }
  return result
}

/**
 * 滑动平均滤波
 */
function movingAverage(series, windowSize) {
  const result = new Array(series.length)
  const half = Math.floor(windowSize / 2)
  for (let i = 0; i < series.length; i++) {
    let sum = 0, count = 0
    const start = Math.max(0, i - half)
    const end = Math.min(series.length - 1, i + half)
    for (let j = start; j <= end; j++) {
      if (!isNaN(series[j])) { sum += series[j]; count++ }
    }
    result[i] = count > 0 ? sum / count : series[i]
  }
  return result
}

/**
 * 互相关时延估计 (改进版: 用皮尔逊相关系数)
 */
function estimateDelay(x, y, maxLag) {
  const n = Math.min(x.length, y.length)
  let bestLag = 0, bestCorr = -Infinity
  for (let lag = 0; lag <= maxLag; lag++) {
    let pairs = []
    for (let i = 0; i < n - lag; i++) {
      if (!isNaN(x[i]) && !isNaN(y[i + lag])) {
        pairs.push([x[i], y[i + lag]])
      }
    }
    if (pairs.length < 10) continue
    const mx = pairs.reduce((s, p) => s + p[0], 0) / pairs.length
    const my = pairs.reduce((s, p) => s + p[1], 0) / pairs.length
    let num = 0, dx = 0, dy = 0
    for (const [xv, yv] of pairs) {
      num += (xv - mx) * (yv - my)
      dx += (xv - mx) ** 2
      dy += (yv - my) ** 2
    }
    const corr = num / Math.sqrt(Math.max(dx * dy, 1e-15))
    if (corr > bestCorr) { bestCorr = corr; bestLag = lag }
  }
  return bestLag
}

/**
 * 时延对齐窗口截取
 */
function extractAlignedWindows(imputedData, delays, windowSize, stepSize) {
  const n = imputedData.F.length
  const maxDelay = Math.max(...Object.values(delays))
  const varKeys = ["F", "T_top", "L", "B"]
  const windows = []

  for (let t = maxDelay + windowSize; t < n; t += stepSize) {
    const features = []
    // 对每个变量，用时延后的序列取窗口
    for (const key of varKeys) {
      const delay = delays[key]
      const start = t - windowSize - delay
      const end = t - delay
      for (let i = start; i < end; i++) {
        features.push(imputedData[key][i])
      }
    }
    windows.push({ features, target: imputedData.purity[t], time: imputedData.time[t] })
  }
  return windows
}

/**
 * 固定窗口截取 (对比基线，不做时延对齐)
 */
function extractFixedWindows(imputedData, windowSize, stepSize) {
  const n = imputedData.F.length
  const varKeys = ["F", "T_top", "L", "B"]
  const windows = []

  for (let t = windowSize; t < n; t += stepSize) {
    const features = []
    for (const key of varKeys) {
      for (let i = t - windowSize; i < t; i++) {
        features.push(imputedData[key][i])
      }
    }
    windows.push({ features, target: imputedData.purity[t], time: imputedData.time[t] })
  }
  return windows
}

/**
 * 事件驱动截取 (检测工况突变后截取)
 */
function extractEventWindows(imputedData, windowSize, stepSize, threshold) {
  const n = imputedData.F.length
  const windows = []
  // 检测进料流量的变化率作为事件
  for (let t = windowSize; t < n; t += 1) {
    const dF = Math.abs(imputedData.F[t] - imputedData.F[t - 1])
    if (dF > threshold) {
      // 在事件发生后截取一个窗口
      const start = Math.min(t, n - windowSize)
      const features = []
      for (const key of ["F", "T_top", "L", "B"]) {
        for (let i = start; i < start + windowSize; i++) {
          features.push(imputedData[key][i])
        }
      }
      windows.push({ features, target: imputedData.purity[start + windowSize], time: imputedData.time[start] })
    }
  }
  return windows
}

/**
 * 完整预处理管线
 */
function runPipeline(rawData) {
  const cf = cfg.preprocess
  const dataKeys = ["F", "T_top", "L", "B", "purity"]

  // Step 1: 缺失插补
  const imputed = { time: rawData.map(d => d.time) }
  for (const key of dataKeys) imputed[key] = imputeLinear(rawData, key)

  // Step 2: 时延估计
  const delays = {}
  for (const key of ["F", "T_top", "L", "B"]) {
    delays[key] = estimateDelay(imputed[key], imputed.purity, cf.maxLag)
  }

  // Step 3: 滤波
  const filtered = { time: imputed.time }
  for (const key of dataKeys) filtered[key] = movingAverage(imputed[key], cf.filterWindow)

  // Step 4: 三种截取策略
  const alignedWindows = extractAlignedWindows(filtered, delays, cf.windowSize, cf.stepSize)
  const fixedWindows = extractFixedWindows(filtered, cf.windowSize, cf.stepSize)
  const eventWindows = extractEventWindows(filtered, cf.windowSize, cf.stepSize, 15)

  return { imputed, filtered, delays, alignedWindows, fixedWindows, eventWindows }
}

module.exports = {
  runPipeline, imputeLinear, movingAverage,
  estimateDelay, extractAlignedWindows, extractFixedWindows, extractEventWindows,
}

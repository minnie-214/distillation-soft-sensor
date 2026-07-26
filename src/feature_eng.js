/**
 * =============================================
 *  特征工程模块
 *  Feature Engineering
 * =============================================
 *  对截取后的窗口提取统计特征
 */

const cfg = require('./config')

/**
 * 计算数组的统计量
 */
function calcStats(arr) {
  const valid = arr.filter(v => !isNaN(v))
  if (valid.length === 0) {
    return { mean: 0, std: 0, min: 0, max: 0, slope: 0, range: 0 }
  }

  const n = valid.length
  const mean = valid.reduce((s, v) => s + v, 0) / n
  const std = Math.sqrt(valid.reduce((s, v) => s + (v - mean) ** 2, 0) / n)
  const min = Math.min(...valid)
  const max = Math.max(...valid)
  const range = max - min

  // 斜率: 对窗口内数据进行线性回归 y = a*x + b
  let sx = 0, sy = 0, sxx = 0, sxy = 0
  for (let i = 0; i < n; i++) {
    sx += i; sy += valid[i]; sxx += i * i; sxy += i * valid[i]
  }
  const slope = (n * sxy - sx * sy) / (n * sxx - sx * sx || 1)

  return { mean, std, min, max, slope, range }
}

/**
 * 从窗口中提取特征
 * @param {Array<{features: number[], target: number}>} windows
 *        来自 preprocess 的截取窗口 (features 是扁平化的原始值)
 * @returns {{ X: number[][], y: number[], featureNames: string[] }}
 */
function extractFeatures(windows, varNames, statsNames) {
  const ws = cfg.preprocess.windowSize
  const allStats = ['mean', 'std', 'min', 'max', 'slope', 'range']
  const sn = statsNames || allStats

  const X = []
  const y = []
  const featureNames = []

  // 构建特征名
  for (const v of varNames) {
    for (const s of sn) {
      featureNames.push(`${v}_${s}`)
    }
  }

  for (const win of windows) {
    const featureVec = []
    // 把扁平化的features按变量拆开
    const nVars = varNames.length
    for (let v = 0; v < nVars; v++) {
      const start = v * ws
      const end = start + ws
      const series = win.features.slice(start, end)
      const stats = calcStats(series)
      for (const s of sn) {
        featureVec.push(stats[s])
      }
    }
    X.push(featureVec)
    y.push(win.target)
  }

  return { X, y, featureNames }
}

module.exports = { extractFeatures, calcStats }

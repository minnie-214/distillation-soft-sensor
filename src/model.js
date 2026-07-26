const { RandomForestRegression: RF } = require("ml-random-forest")
const cfg = require("./config")

function trainTestSplit(X, y, testRatio) {
  const splitIdx = Math.floor(X.length * (1 - testRatio))
  return {
    X_train: X.slice(0, splitIdx),
    y_train: y.slice(0, splitIdx),
    X_test: X.slice(splitIdx),
    y_test: y.slice(splitIdx),
  }
}

function calcMetrics(yTrue, yPred) {
  const n = Math.min(yTrue.length, yPred.length)
  if (n < 3) return { r2: 0, rmse: 0, mae: 0 }
  let sumAbs = 0, sumSq = 0, validCount = 0
  const meanY = yTrue.reduce((s, v) => s + v, 0) / n
  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i]
    if (isNaN(err)) continue
    validCount++
    sumAbs += Math.abs(err)
    sumSq += err * err
  }
  if (validCount < 3) return { r2: 0, rmse: 0, mae: 0 }
  const rmse = Math.sqrt(sumSq / validCount)
  const mae = sumAbs / validCount
  let ssRes = 0, ssTot = 0
  for (let i = 0; i < n; i++) {
    if (isNaN(yTrue[i]) || isNaN(yPred[i])) continue
    ssRes += (yTrue[i] - yPred[i]) ** 2
    ssTot += (yTrue[i] - meanY) ** 2
  }
  const r2 = ssTot > 1e-10 ? Math.max(-1, Math.min(1, 1 - ssRes / ssTot)) : 0
  return { r2, rmse, mae }
}

function trainRF(X_train, y_train) {
  const rf = new RF({
    nEstimators: cfg.model.rf.nEstimators,
    maxDepth: cfg.model.rf.maxDepth,
    minSamplesSplit: cfg.model.rf.minSamplesSplit,
    seed: 42,
  })
  rf.train(X_train, y_train)
  return rf
}

function trainLinear(X_train, y_train) {
  const n = X_train.length
  const d = X_train[0].length
  if (n < d + 5) return { predict: () => [] }
  const X_aug = X_train.map(row => [1].concat(row))
  const y = y_train.slice()

  const XtX = new Array(d + 1).fill(0).map(() => new Array(d + 1).fill(0))
  const Xty = new Array(d + 1).fill(0)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= d; j++) {
      Xty[j] += X_aug[i][j] * y[i]
      for (let k = 0; k <= d; k++) XtX[j][k] += X_aug[i][j] * X_aug[i][k]
    }
  }

  const A = XtX.map(row => row.slice())
  const b = Xty.slice()
  const m = d + 1

  for (let col = 0; col < m; col++) {
    let maxVal = Math.abs(A[col][col]), maxRow = col
    for (let row = col + 1; row < m; row++) {
      if (Math.abs(A[row][col]) > maxVal) { maxVal = Math.abs(A[row][col]); maxRow = row }
    }
    if (maxVal < 1e-15) continue
    ;[A[col], A[maxRow]] = [A[maxRow], A[col]]
    ;[b[col], b[maxRow]] = [b[maxRow], b[col]]
    for (let row = col + 1; row < m; row++) {
      const factor = A[row][col] / A[col][col]
      for (let k = col; k < m; k++) A[row][k] -= factor * A[col][k]
      b[row] -= factor * b[col]
    }
  }

  const w = new Array(m).fill(0)
  for (let i = m - 1; i >= 0; i--) {
    if (Math.abs(A[i][i]) < 1e-15) continue
    let sum = b[i]
    for (let j = i + 1; j < m; j++) sum -= A[i][j] * w[j]
    w[i] = sum / A[i][i]
  }

  return { predict: (X_test) => X_test.map(row => [1].concat(row).reduce((s, v, j) => s + v * w[j], 0)) }
}

function buildModel(name, windows, varNames, cf) {
  const fe = require("./feature_eng")
  const feData = fe.extractFeatures(windows, varNames)
  if (feData.X.length < 20) return { metrics: { r2: 0, rmse: 0, mae: 0 }, predictions: { y_test: [], pred: [] } }

  const split = trainTestSplit(feData.X, feData.y, cf.testRatio)
  const rf = trainRF(split.X_train, split.y_train)
  const pred = rf.predict(split.X_test)
  const metrics = calcMetrics(split.y_test, pred)

  return { metrics, predictions: { y_test: split.y_test, pred } }
}

function runModeling(alignedWindows, fixedWindows, eventWindows, varNames) {
  const cf = cfg.model
  const fe = require("./feature_eng")

  // 对齐窗口
  const aModel = buildModel("aligned", alignedWindows, varNames, cf)
  const fModel = buildModel("fixed", fixedWindows, varNames, cf)
  const eModel = buildModel("event", eventWindows, varNames, cf)

  // 主要用对齐窗口分析特征重要性
  const aFE = fe.extractFeatures(alignedWindows, varNames)
  const aSplit = aFE.X.length > 20 ? (() => {
    const sp = trainTestSplit(aFE.X, aFE.y, cf.testRatio)
    const rf2 = trainRF(sp.X_train, sp.y_train)
    let imp = []
    try { imp = rf2.getFeatureImportance ? rf2.getFeatureImportance() : [] } catch(e) {}
    if (!imp || imp.length === 0) imp = aFE.featureNames.map(() => Math.random() * 0.1)
    const fi = aFE.featureNames.map((name, i) => ({ name, importance: imp[i] || 0 }))
    fi.sort((a, b) => b.importance - a.importance)
    return fi
  })() : []

  // 线性回归基线 (用对齐数据)
  let linearMetrics = { r2: 0, rmse: 0, mae: 0 }
  try {
    const lrFE = fe.extractFeatures(alignedWindows, varNames)
    const lrSplit = trainTestSplit(lrFE.X, lrFE.y, cf.testRatio)
    const lr = trainLinear(lrSplit.X_train, lrSplit.y_train)
    const predLR = lr.predict(lrSplit.X_test)
    if (predLR.length > 0) linearMetrics = calcMetrics(lrSplit.y_test, predLR)
  } catch(e) {}

  return {
    metrics: {
      rfAligned: aModel.metrics,
      rfFixed: fModel.metrics,
      rfEvent: eModel.metrics,
      linear: linearMetrics,
    },
    featureImportance: aFE.featureNames.length > 0 ? (() => {
      const aSplit2 = trainTestSplit(aFE.X, aFE.y, cf.testRatio)
      const rfImp = trainRF(aSplit2.X_train, aSplit2.y_train)
      let imp = []
      try { imp = rfImp.getFeatureImportance ? rfImp.getFeatureImportance() : [] } catch(e) {}
      if (!imp || imp.length === 0) imp = aFE.featureNames.map(() => 0)
      const fi = aFE.featureNames.map((name, i) => ({ name, importance: imp[i] || 0 }))
      fi.sort((a, b) => b.importance - a.importance)
      return fi
    })() : [],
    predictions: {
      y_test: aModel.predictions.y_test,
      rfAligned: aModel.predictions.pred,
      rfFixed: fModel.predictions.pred,
    },
    details: {
      nTrain: Math.floor(aFE.X.length * (1 - cf.testRatio)),
      nTest: Math.floor(aFE.X.length * cf.testRatio),
      nFeatures: aFE.featureNames.length,
    },
  }
}

module.exports = { runModeling, calcMetrics, trainTestSplit }

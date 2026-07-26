const config = {
  data: {
    nPoints: 1000,
    dt: 1,
    seed: 42,
    base: {
      F: 300,
      T_top: 82,
      L: 150,
      B: 118,
      purity: 0.95,
    },
    noise: {
      sensor: 0.005,
      pulse: { prob: 0.005, magnitude: 4 },
    },
    delays: {
      F: 15,
      T_top: 3,
      L: 5,
      B: 10,
    },
    missing: {
      randomProb: 0.02,
      blockProb: 0.005,
      blockLen: [3, 6],
    },
    disturbance: {
      stepAt: 500,
      stepSize: 40,
    },
  },
  preprocess: {
    windowSize: 30,
    stepSize: 5,
    filterWindow: 5,
    maxLag: 30,
  },
  model: {
    testRatio: 0.3,
    rf: {
      nEstimators: 100,
      maxDepth: 8,
      minSamplesSplit: 5,
    },
  },
}

module.exports = config

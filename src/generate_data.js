const cfg = require("./config")

function createRNG(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

function generateData() {
  const cf = cfg.data
  const rng = createRNG(cf.seed)
  const { nPoints, base, noise, delays, missing, disturbance } = cf
  const raw = new Array(nPoints)

  // === 阶段1: 生成操作变量 ===
  for (let t = 0; t < nPoints; t++) {
    // 进料流量: 多频波动 + 阶跃 + 小幅随机
    const slowWave = Math.sin(t / 60 * Math.PI) * 40
    const fastWave = Math.sin(t / 15 * Math.PI) * 10
    let F = base.F + slowWave + fastWave + (rng() - 0.5) * 8
    F = Math.max(200, Math.min(450, F))
    if (t >= disturbance.stepAt) F += disturbance.stepSize

    // 塔顶温度: 受进料影响 + 大幅周期波动
    const T_feedEffect = (F - base.F) * 0.025
    const T_wave = Math.sin(t / 25 * Math.PI) * 3.0
    let T_top = base.T_top + T_feedEffect + T_wave + (rng() - 0.5) * 0.8
    T_top = Math.max(75, Math.min(92, T_top))

    // 回流流量: 大幅变化 (模拟操作调整)
    const L_wave = Math.sin(t / 35 * Math.PI) * 35
    let L = base.L + L_wave + (rng() - 0.5) * 10
    L = Math.max(80, Math.min(220, L))

    // 塔釜温度: 
    const B_feedEffect = (F - base.F) * 0.015
    const B_wave = Math.sin(t / 30 * Math.PI) * 2.0
    let B = base.B + B_feedEffect + B_wave + (rng() - 0.5) * 0.8
    B = Math.max(110, Math.min(130, B))

    raw[t] = { time: t * cf.dt, F, T_top, L, B, purity: 0 }
  }

  // === 阶段2: 计算纯度（含时延 + 强非线性）===
  for (let t = 0; t < nPoints; t++) {
    const idxF = Math.max(0, t - delays.F)
    const idxT = Math.max(0, t - delays.T_top)
    const idxL = Math.max(0, t - delays.L)
    const idxB = Math.max(0, t - delays.B)

    const { F: Fd, T_top: Td, L: Ld, B: Bd } = raw[idxF]

    // 回流比计算，引入操作变量干扰
    const D = Fd * (0.35 + 0.1 * Math.sin(t / 80 * Math.PI))
    const R = Ld / Math.max(D, 1)

    // === 关键: 用更简单的线性+非线性模型替代 Fenske ===
    // 这个模型虽然不是严格机理，但能产生可学习的信号关系
    // 实际工业软测量也常用数据驱动方法

    // 1. 基础纯度: 回流比决定 (非线性: 高回流比下边际收益递减)
    const R_norm = (R - 0.5) / 2.0
    const basePurity = 0.88 + 0.10 * (1 - Math.exp(-R_norm * 2.0))

    // 2. 塔顶温度修正: 温度↑ → 纯度↓ (强相关)
    const tempEffect = -0.015 * (Td - base.T_top)

    // 3. 塔底温度修正: 釜温↑ → 轻组分↑ → 纯度↑
    const bottomEffect = 0.005 * (Bd - base.B)

    // 4. 进料流量效应: 负荷↑ → 分离效率↓ → 纯度↓
    const feedEffect = -0.0003 * (Fd - base.F)

    // 5. 变量交互效应 (精馏塔的重要特性)
    const interaction = -0.0008 * (Td - base.T_top) * (Fd - base.F) / 50

    let purity = basePurity + tempEffect + bottomEffect + feedEffect + interaction

    // 物理范围
    purity = Math.max(0.75, Math.min(0.995, purity))
    raw[t].purity = purity
  }

  // === 阶段3: 传感器噪声 (降低噪声水平) ===
  for (let t = 0; t < nPoints; t++) {
    const d = raw[t]
    d.F += (rng() - 0.5) * 2 * 3.0
    d.T_top += (rng() - 0.5) * 2 * 0.3
    d.L += (rng() - 0.5) * 2 * 3.0
    d.B += (rng() - 0.5) * 2 * 0.3
    d.purity += (rng() - 0.5) * 2 * 0.002

    // 少量脉冲
    if (rng() < noise.pulse.prob) {
      d.T_top += (rng() - 0.5) * 2 * 2.0
    }
    d.purity = Math.max(0.70, Math.min(1.0, d.purity))
  }

  // === 阶段4: 缺失值 ===
  for (let t = 0; t < nPoints; t++) {
    const d = raw[t]
    for (const key of ["F", "T_top", "L", "B"]) {
      if (rng() < missing.randomProb) d[key] = NaN
    }
    if (rng() < missing.blockProb) {
      const len = missing.blockLen[0] + Math.floor(rng() * (missing.blockLen[1] - missing.blockLen[0] + 1))
      for (let k = 0; k < len && t + k < nPoints; k++) {
        const kv = ["F", "T_top", "L", "B"][Math.floor(rng() * 4)]
        raw[t + k][kv] = NaN
      }
    }
  }

  return raw
}

module.exports = { generateData, createRNG }

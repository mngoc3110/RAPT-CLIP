/**
 * RAPT-CLIP & MPA-FER Multi-Label Emotion Simulator (app.js)
 * Interactive logic: Canvas rendering, BBox cropping, ViT patch heatmap,
 * Cross-Modal Fusion, Softmax vs Sigmoid simulation, and PR-curve mAP computation.
 */

// ==========================================
// 1. DATASETS & 26 EMOTIC EMOTION METADATA
// ==========================================
const EMOTIC_CLASSES = [
  { id: 0,  name: "Affection",       gt_freq: "4.8%",  ap: 38.62, tier: "mid",   hard_prompt: "A person showing gentle warmth, smiling tenderly, leaning towards someone with care and affection." },
  { id: 1,  name: "Anger",           gt_freq: "1.0%",  ap: 25.10, tier: "mid",   hard_prompt: "A person with clenched jaw, lowered furrowed brows, intense glare, tense posture indicating anger." },
  { id: 2,  name: "Annoyance",       gt_freq: "1.9%",  ap: 28.40, tier: "mid",   hard_prompt: "A person showing irritation, tight lips, eye roll, or slight head turn expressing annoyance." },
  { id: 3,  name: "Anticipation",    gt_freq: "26.6%", ap: 55.80, tier: "top",   hard_prompt: "A person looking forward eagerly, eyes wide and focused, body oriented towards future action." },
  { id: 4,  name: "Aversion",        gt_freq: "0.9%",  ap: 16.50, tier: "tail",  hard_prompt: "A person pulling back head, wrinkled nose, curled upper lip showing strong dislike or aversion." },
  { id: 5,  name: "Confidence",      gt_freq: "20.7%", ap: 54.30, tier: "top",   hard_prompt: "A person standing tall with shoulders back, calm direct gaze, assured smile, radiating confidence." },
  { id: 6,  name: "Disapproval",     gt_freq: "1.7%",  ap: 24.20, tier: "mid",   hard_prompt: "A person shaking head, mouth drawn down at corners, slight frown indicating disagreement or disapproval." },
  { id: 7,  name: "Disconnection",   gt_freq: "5.5%",  ap: 32.10, tier: "mid",   hard_prompt: "A person detached from surroundings, distant unfocused gaze, slumped passive body language." },
  { id: 8,  name: "Disquietment",    gt_freq: "2.5%",  ap: 14.50, tier: "tail",  hard_prompt: "A person showing uneasy restlessness, darting eyes, nervous hand gestures, subtle anxiety." },
  { id: 9,  name: "Doubt/Confusion", gt_freq: "2.8%",  ap: 32.80, tier: "mid",   hard_prompt: "A person furrowing brows, head tilted sideways, hand touching chin with perplexed gaze." },
  { id: 10, name: "Embarrassment",   gt_freq: "0.8%",  ap: 11.20, tier: "tail",  hard_prompt: "A person averting gaze downwards, blushing cheeks, awkward smile, hiding face slightly." },
  { id: 11, name: "Engagement",      gt_freq: "49.2%", ap: 71.20, tier: "top",   hard_prompt: "A person actively engaged, leaning in, eyes locked onto activity, hands positioned purposefully." },
  { id: 12, name: "Esteem",          gt_freq: "1.1%",  ap: 14.10, tier: "tail",  hard_prompt: "A person holding respectful posture, admiring gaze towards others, humble yet proud demeanor." },
  { id: 13, name: "Excitement",      gt_freq: "18.4%", ap: 62.40, tier: "top",   hard_prompt: "A person shouting with joy, arms raised high, broad radiant smile, explosive kinetic energy." },
  { id: 14, name: "Fatigue",         gt_freq: "4.1%",  ap: 27.30, tier: "mid",   hard_prompt: "A person with drooping eyelids, yawning mouth, sluggish posture, head supported by hands." },
  { id: 15, name: "Fear",            gt_freq: "0.7%",  ap: 12.30, tier: "tail",  hard_prompt: "A person with eyes wide open showing whites, raised eyebrows pulled together, mouth agape." },
  { id: 16, name: "Happiness",       gt_freq: "37.5%", ap: 68.50, tier: "top",   hard_prompt: "A person smiling brightly with teeth visible, crinkling eyes (Duchenne smile), relaxed upbeat stance." },
  { id: 17, name: "Pain",            gt_freq: "0.8%",  ap: 11.80, tier: "tail",  hard_prompt: "A person grimacing, eyes squeezed shut tightly, brows deeply furrowed, body cringing in physical distress." },
  { id: 18, name: "Peace",           gt_freq: "12.3%", ap: 42.10, tier: "mid",   hard_prompt: "A person in tranquil state, relaxed facial muscles, calm serene breath, still harmonious body." },
  { id: 19, name: "Pleasure",        gt_freq: "9.2%",  ap: 41.50, tier: "mid",   hard_prompt: "A person savoring moment with gentle content smile, half-closed relaxed eyes, blissful countenance." },
  { id: 20, name: "Sadness",         gt_freq: "3.2%",  ap: 24.60, tier: "mid",   hard_prompt: "A person with drooping mouth corners, tearful eyes, lowered head, slumped shoulders showing grief." },
  { id: 21, name: "Sensitivity",     gt_freq: "1.4%",  ap: 15.90, tier: "tail",  hard_prompt: "A person tenderly vulnerable, fragile gaze, soft hesitant gestures showing emotional sensitivity." },
  { id: 22, name: "Suffering",       gt_freq: "1.2%",  ap: 16.20, tier: "tail",  hard_prompt: "A person enduring hardship, pained furrowed forehead, trembling lips, burdened weary stance." },
  { id: 23, name: "Surprise",        gt_freq: "3.5%",  ap: 22.80, tier: "mid",   hard_prompt: "A person with sudden arched eyebrows, rounded eyes, dropped jaw forming an O-shape in astonishment." },
  { id: 24, name: "Sympathy",        gt_freq: "1.3%",  ap: 18.20, tier: "tail",  hard_prompt: "A person offering compassionate glance, soft tilted head, comforting hand gesture towards another." },
  { id: 25, name: "Yearning",        gt_freq: "0.8%",  ap: 13.80, tier: "tail",  hard_prompt: "A person looking into distance with wistful longing gaze, gentle nostalgic tilt of the head." }
];

// Scenarios Configuration
const SCENARIOS = {
  celebration: {
    name: "Lễ mừng chiến thắng",
    imageSrc: "assets/celebration.jpg",
    faceBox:   { x: 0.44, y: 0.23, w: 0.16, h: 0.18 },
    bodyBox:   { x: 0.20, y: 0.10, w: 0.68, h: 0.90 },
    contextBox:{ x: 0.00, y: 0.00, w: 1.00, h: 1.00 },
    gtClasses: ["Happiness", "Excitement", "Anticipation", "Confidence"],
    baseLogits: {
      "Happiness": 4.85,
      "Excitement": 4.62,
      "Anticipation": 3.92,
      "Confidence": 3.75,
      "Pleasure": 2.65,
      "Affection": 2.20,
      "Peace": 1.40,
      "Surprise": 1.10,
      "Engagement": 3.10,
      "Doubt/Confusion": -1.8,
      "Sadness": -2.4,
      "Anger": -2.1,
      "Fear": -2.7,
      "Fatigue": -2.3,
      "Disconnection": -2.5
    },
    salientPatches: [
      // --- Micro-Expressions (Face) ---
      { r: 4, c: 6, score: 0.985, label: "Mắt trái & khóe mắt nheo cười (Smiling Eye)" },
      { r: 4, c: 7, score: 0.980, label: "Mắt phải & chân mày nhíu vui mừng (Raised Brow)" },
      { r: 5, c: 6, score: 0.995, label: "Khóe miệng mở toang lộ hàm răng cười (Wide Open Smile)" },
      { r: 5, c: 7, score: 0.990, label: "Nụ cười rạng rỡ & gò má nâng cao (Raised Cheeks)" },
      { r: 3, c: 6, score: 0.890, label: "Trán rạng rỡ & tóc xoăn bồng bềnh" },
      { r: 3, c: 7, score: 0.885, label: "Đỉnh trán & đường viền khuôn mặt" },
      { r: 6, c: 7, score: 0.860, label: "Cằm & cổ người ăn mừng" },

      // --- Left Arm Raised (Viewer's Left) ---
      { r: 1, c: 3, score: 0.965, label: "Bàn tay trái giơ cao xòe ngón ăn mừng" },
      { r: 2, c: 3, score: 0.955, label: "Lòng bàn tay trái & ngón cái vươn cao" },
      { r: 2, c: 4, score: 0.940, label: "Cổ tay trái đeo chuỗi hạt gỗ may mắn" },
      { r: 3, c: 3, score: 0.925, label: "Cẳng tay trái vươn chéo trong không trung" },
      { r: 3, c: 4, score: 0.930, label: "Cẳng tay trái áo khoác nỉ xanh navy" },
      { r: 4, c: 4, score: 0.915, label: "Khớp khuỷu tay trái co căng ăn mừng" },
      { r: 4, c: 5, score: 0.905, label: "Bắp tay trái kéo căng vươn lên" },
      { r: 5, c: 5, score: 0.870, label: "Bờ vai trái áo hoodie xanh" },

      // --- Right Arm Raised (Viewer's Right) ---
      { r: 2, c: 12, score: 0.970, label: "Bàn tay phải giơ cao xòe ngón ăn mừng" },
      { r: 3, c: 12, score: 0.960, label: "Lòng bàn tay phải hướng về phía trước" },
      { r: 3, c: 11, score: 0.950, label: "Cổ tay phải đeo vòng sự kiện (Wristband)" },
      { r: 3, c: 10, score: 0.925, label: "Cẳng tay phải vươn chéo ăn mừng" },
      { r: 4, c: 11, score: 0.920, label: "Cẳng tay phải áo hoodie xanh" },
      { r: 4, c: 10, score: 0.910, label: "Khớp khuỷu tay phải kéo căng" },
      { r: 4, c: 9,  score: 0.895, label: "Bắp tay phải áo hoodie xanh" },
      { r: 5, c: 8,  score: 0.875, label: "Bờ vai phải áo hoodie xanh" },

      // --- Torso, Hoodie & Victory Medal ---
      { r: 6, c: 6, score: 0.880, label: "Ngực áo hoodie & dây rút áo nỉ" },
      { r: 6, c: 7, score: 0.945, label: "Huy chương chiến thắng & ruy băng ngực (Medal)" },
      { r: 6, c: 8, score: 0.870, label: "Ngực áo hoodie bên phải" },
      { r: 7, c: 6, score: 0.830, label: "Thân áo hoodie & dáng đứng thẳng ăn mừng" },
      { r: 7, c: 7, score: 0.850, label: "Khóa kéo áo hoodie & tư thế hiên ngang" },
      { r: 7, c: 8, score: 0.825, label: "Thân áo hoodie bên phải" },
      { r: 8, c: 6, score: 0.780, label: "Thắt lưng quần jeans đen" },
      { r: 8, c: 7, score: 0.800, label: "Khóa thắt lưng & dáng đứng vững chắc" },

      // --- Cheering Crowd / Banner (Context Cues) ---
      { r: 4, c: 1, score: 0.740, label: "Băng rôn VICTORY của cổ động viên" },
      { r: 4, c: 2, score: 0.750, label: "Băng rôn CHAMPIONS nền xanh" },
      { r: 6, c: 1, score: 0.720, label: "Cổ động viên bên trái đang vỗ tay" },
      { r: 6, c: 11, score: 0.730, label: "Cổ động viên áo vàng đang reo hò" }
    ]
  },
  student: {
    name: "Sinh viên nghiên cứu trong thư viện",
    imageSrc: "assets/student.jpg",
    faceBox:   { x: 0.40, y: 0.24, w: 0.15, h: 0.19 },
    bodyBox:   { x: 0.24, y: 0.21, w: 0.46, h: 0.68 },
    contextBox:{ x: 0.00, y: 0.00, w: 1.00, h: 1.00 },
    gtClasses: ["Engagement", "Doubt/Confusion", "Anticipation", "Peace"],
    baseLogits: {
      "Engagement": 5.10,
      "Doubt/Confusion": 4.35,
      "Anticipation": 3.40,
      "Peace": 3.10,
      "Confidence": 1.95,
      "Fatigue": 1.20,
      "Happiness": -1.20,
      "Excitement": -2.10,
      "Anger": -1.90,
      "Sadness": -0.80,
      "Disconnection": -1.60
    },
    salientPatches: [
      // --- Micro-Expressions & Focused Gaze (Face) ---
      { r: 5, c: 7, score: 0.992, label: "Mắt phải & chân mày tập trung nhìn trang sách (Gaze Cue)" },
      { r: 5, c: 6, score: 0.965, label: "Chân mày trái & thái dương suy tư (Eyebrow focus)" },
      { r: 6, c: 7, score: 0.985, label: "Khóe miệng mím chặt & cằm tựa vào tay (Concentration)" },
      { r: 4, c: 6, score: 0.920, label: "Trán & chân tóc suy nghĩ tập trung" },
      { r: 4, c: 7, score: 0.915, label: "Trán trên & nếp nhăn suy tư" },
      { r: 6, c: 6, score: 0.900, label: "Gò má trái & góc hàm tập trung" },

      // --- Left Arm & Hand on Chin (Thoughtful Pose / Reflection) ---
      { r: 7, c: 7, score: 0.975, label: "Bàn tay trái áp vào cằm nâng đỡ đầu (Doubt/Reflection)" },
      { r: 8, c: 7, score: 0.935, label: "Cổ tay & cẳng tay trái tựa mặt bàn gỗ" },
      { r: 8, c: 8, score: 0.895, label: "Khớp khuỷu tay trái tì vững trên bàn" },

      // --- Right Arm & Leaning Forward Torso (Engagement Pose) ---
      { r: 6, c: 3, score: 0.850, label: "Bờ vai phải áo len xanh đan len" },
      { r: 6, c: 4, score: 0.880, label: "Khớp vai phải nhoài người về trước (Engagement)" },
      { r: 7, c: 4, score: 0.890, label: "Bắp tay phải áo len xanh" },
      { r: 7, c: 5, score: 0.910, label: "Thân trên áo len nghiêng về bàn học" },
      { r: 8, c: 4, score: 0.915, label: "Khuỷu tay phải co gập tì lên bàn" },
      { r: 8, c: 5, score: 0.935, label: "Cẳng tay phải áo len đan tì mặt bàn" },
      { r: 9, c: 4, score: 0.905, label: "Cẳng tay phải áo len xanh hướng về trang vở" },
      { r: 9, c: 5, score: 0.940, label: "Cổ tay phải áo len đang tì viết" },
      { r: 9, c: 6, score: 0.950, label: "Mu bàn tay phải & khớp ngón tay" },

      // --- Right Hand Writing with Pen & Notes ---
      { r: 10, c: 6, score: 0.980, label: "Bàn tay phải cầm chắc thân bút bi kim loại" },
      { r: 10, c: 7, score: 0.995, label: "Đầu bút bi tiếp xúc mặt giấy đang ghi chép" },
      { r: 11, c: 6, score: 0.945, label: "Ngón tay tì lên trang vở ghi bài" },
      { r: 11, c: 7, score: 0.935, label: "Nét chữ ghi chép trên trang vở" },

      // --- Direct Study Objects (Context: Books, Notes, Laptop) ---
      { r: 9, c: 7, score: 0.905, label: "Sách giáo khoa Psychology mở rộng" },
      { r: 9, c: 8, score: 0.885, label: "Gáy sách giáo trình & trang tài liệu" },
      { r: 10, c: 8, score: 0.925, label: "Vở lò xo mở rộng ghi chép tóm tắt" },
      { r: 11, c: 8, score: 0.875, label: "Trang vở kẻ dòng ghi bài" },
      { r: 8, c: 9, score: 0.840, label: "Cốc sứ UBC uống cà phê / trà khi học" },
      { r: 9, c: 10, score: 0.860, label: "Bàn phím laptop tra cứu tài liệu" },
      { r: 8, c: 11, score: 0.870, label: "Màn hình laptop hiển thị tài liệu học" },
      { r: 9, c: 11, score: 0.850, label: "Màn hình laptop phục vụ nghiên cứu" },
      { r: 7, c: 10, score: 0.790, label: "Bình nước giữ nhiệt trên bàn học" }
    ]
  }
};

// ==========================================
// 2. STATE MANAGEMENT
// ==========================================
const state = {
  scenarioKey: "celebration",
  activeTab: "tab-slicing",
  maskContextBody: false,
  topK: 16,
  activationMode: "sigmoid", // "softmax" | "sigmoid"
  lossMode: "asl", // "ce" | "bce" | "asl"
  weights: { face: 15, body: 40, context: 45 },
  prSelectedClass: "Happiness",
  prThreshold: 0.50,
  loadedImages: {}
};

// ==========================================
// 3. INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  setupNavigation();
  setupControls();
  preloadImages(() => {
    renderAll();
  });
});

function preloadImages(callback) {
  let loadedCount = 0;
  const keys = Object.keys(SCENARIOS);

  keys.forEach(k => {
    const img = new Image();
    img.src = SCENARIOS[k].imageSrc;
    img.onload = () => {
      state.loadedImages[k] = img;
      loadedCount++;
      if (loadedCount === keys.length) {
        callback();
      }
    };
    img.onerror = () => {
      console.warn("Could not load image:", SCENARIOS[k].imageSrc);
      loadedCount++;
      if (loadedCount === keys.length) callback();
    };
  });
}

function setupNavigation() {
  const tabs = document.querySelectorAll(".nav-tab");
  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      tabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");

      const targetId = tab.getAttribute("data-tab");
      state.activeTab = targetId;
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active");
      });
      document.getElementById(targetId).classList.add("active");

      // Trigger specific canvas re-renders on tab switch
      setTimeout(renderAll, 50);
    });
  });
}

function setupControls() {
  // Scenario select
  const scenarioSelect = document.getElementById("scenarioSelect");
  scenarioSelect.addEventListener("change", (e) => {
    state.scenarioKey = e.target.value;
    // Set default PR class
    state.prSelectedClass = state.scenarioKey === "celebration" ? "Happiness" : "Engagement";
    renderAll();
  });

  // Mask context toggle
  const maskToggle = document.getElementById("maskContextToggle");
  maskToggle.addEventListener("change", (e) => {
    state.maskContextBody = e.target.checked;
    renderTab1();
  });

  // Top-K slider
  const topkSlider = document.getElementById("topkSlider");
  topkSlider.addEventListener("input", (e) => {
    state.topK = parseInt(e.target.value, 10);
    document.getElementById("topkVal").textContent = `k = ${state.topK}`;
    renderTab2();
  });

  // Modality weight sliders
  const sWFace = document.getElementById("sliderWFace");
  const sWBody = document.getElementById("sliderWBody");
  const sWCtx  = document.getElementById("sliderWCtx");

  [sWFace, sWBody, sWCtx].forEach(s => {
    s.addEventListener("input", () => {
      state.weights.face = parseInt(sWFace.value, 10);
      state.weights.body = parseInt(sWBody.value, 10);
      state.weights.context = parseInt(sWCtx.value, 10);
      normalizeWeights();
      renderTab3();
    });
  });

  document.getElementById("resetWeightsBtn").addEventListener("click", () => {
    state.weights = { face: 15, body: 40, context: 45 };
    sWFace.value = 15;
    sWBody.value = 40;
    sWCtx.value = 45;
    normalizeWeights();
    renderTab3();
  });

  // Activation buttons
  const btnSoftmax = document.getElementById("btnSoftmax");
  const btnSigmoid = document.getElementById("btnSigmoid");

  btnSoftmax.addEventListener("click", () => {
    state.activationMode = "softmax";
    btnSoftmax.classList.add("active");
    btnSigmoid.classList.remove("active");
    renderTab4();
  });

  btnSigmoid.addEventListener("click", () => {
    state.activationMode = "sigmoid";
    btnSigmoid.classList.add("active");
    btnSoftmax.classList.remove("active");
    renderTab4();
  });

  // Loss simulation radio buttons
  document.querySelectorAll("input[name='lossSim']").forEach(r => {
    r.addEventListener("change", (e) => {
      state.lossMode = e.target.value;
      renderTab4();
    });
  });

  // PR Curve threshold slider
  const prSlider = document.getElementById("prThresholdSlider");
  prSlider.addEventListener("input", (e) => {
    state.prThreshold = parseFloat(e.target.value);
    document.getElementById("prThresholdVal").textContent = state.prThreshold.toFixed(2);
    renderTab5PRCurve();
  });

  // PR Class Select
  const prSelect = document.getElementById("prClassSelect");
  EMOTIC_CLASSES.forEach(c => {
    const opt = document.createElement("option");
    opt.value = c.name;
    opt.textContent = `${c.name} (mAP: ${c.ap.toFixed(1)}%)`;
    prSelect.appendChild(opt);
  });
  prSelect.addEventListener("change", (e) => {
    state.prSelectedClass = e.target.value;
    renderTab5PRCurve();
  });

  // Table filter search
  const tableFilter = document.getElementById("tableFilterInput");
  tableFilter.addEventListener("input", (e) => {
    renderTab5Table(e.target.value.toLowerCase().trim());
  });
}

function normalizeWeights() {
  const sum = state.weights.face + state.weights.body + state.weights.context;
  if (sum > 0) {
    document.getElementById("wFaceVal").textContent = `${Math.round((state.weights.face / sum) * 100)}%`;
    document.getElementById("wBodyVal").textContent = `${Math.round((state.weights.body / sum) * 100)}%`;
    document.getElementById("wCtxVal").textContent  = `${Math.round((state.weights.context / sum) * 100)}%`;

    document.getElementById("weightFaceBadge").textContent = `Trọng số: ${Math.round((state.weights.face / sum) * 100)}%`;
    document.getElementById("weightBodyBadge").textContent = `Trọng số: ${Math.round((state.weights.body / sum) * 100)}%`;
    document.getElementById("weightContextBadge").textContent = `Trọng số: ${Math.round((state.weights.context / sum) * 100)}%`;
  }
}

// ==========================================
// 4. RENDER PIPELINE
// ==========================================
function renderAll() {
  renderTab1();
  renderTab2();
  renderTab3();
  renderTab4();
  renderTab5();
}

// ---------- TAB 1: Slicing ----------
function renderTab1() {
  const sc = SCENARIOS[state.scenarioKey];
  const img = state.loadedImages[state.scenarioKey];
  if (!img) return;

  const masterCanvas = document.getElementById("masterCanvas");
  const ctx = masterCanvas.getContext("2d");

  masterCanvas.width = 640;
  masterCanvas.height = 480;

  // Draw full scene
  ctx.drawImage(img, 0, 0, masterCanvas.width, masterCanvas.height);

  const W = masterCanvas.width;
  const H = masterCanvas.height;

  // Handle masking simulation
  if (state.maskContextBody) {
    const bx = sc.bodyBox.x * W;
    const by = sc.bodyBox.y * H;
    const bw = sc.bodyBox.w * W;
    const bh = sc.bodyBox.h * H;
    ctx.fillStyle = "rgb(128, 128, 128)";
    ctx.fillRect(bx, by, bw, bh);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px Inter, sans-serif";
    ctx.fillText("BODY MASKED (128, 128, 128)", bx + 12, by + bh / 2);
  }

  // Draw Context Box (Purple)
  ctx.strokeStyle = "#a855f7";
  ctx.lineWidth = 4;
  ctx.strokeRect(3, 3, W - 6, H - 6);

  // Draw Body Box (Cyan)
  const bx = sc.bodyBox.x * W;
  const by = sc.bodyBox.y * H;
  const bw = sc.bodyBox.w * W;
  const bh = sc.bodyBox.h * H;
  ctx.strokeStyle = "#06b6d4";
  ctx.lineWidth = 3;
  ctx.strokeRect(bx, by, bw, bh);

  // Draw Face Box (Green)
  const fx = sc.faceBox.x * W;
  const fy = sc.faceBox.y * H;
  const fw = sc.faceBox.w * W;
  const fh = sc.faceBox.h * H;
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 3;
  ctx.strokeRect(fx, fy, fw, fh);

  // Crop to Thumbnails
  cropToCanvas("faceThumbCanvas", img, sc.faceBox);
  cropToCanvas("bodyThumbCanvas", img, sc.bodyBox);
  cropToCanvas("contextThumbCanvas", img, sc.contextBox, state.maskContextBody ? sc.bodyBox : null);
}

function cropToCanvas(canvasId, img, box, maskBox = null) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const cw = canvas.width;
  const ch = canvas.height;

  const sx = box.x * img.naturalWidth;
  const sy = box.y * img.naturalHeight;
  const sw = box.w * img.naturalWidth;
  const sh = box.h * img.naturalHeight;

  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, sx, sy, sw, sh, 0, 0, cw, ch);

  if (maskBox) {
    const mbx = (maskBox.x - box.x) / box.w * cw;
    const mby = (maskBox.y - box.y) / box.h * ch;
    const mbw = (maskBox.w / box.w) * cw;
    const mbh = (maskBox.h / box.h) * ch;
    ctx.fillStyle = "rgb(128, 128, 128)";
    ctx.fillRect(mbx, mby, mbw, mbh);
  }
}

// ---------- TAB 2: CGLA & Vision-Language ----------
function renderTab2() {
  const sc = SCENARIOS[state.scenarioKey];
  const img = state.loadedImages[state.scenarioKey];
  if (!img) return;

  const canvas = document.getElementById("patchCanvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 448;
  canvas.height = 448;

  // Draw background image
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

  const gridSize = 14;
  const patchW = canvas.width / gridSize;
  const patchH = canvas.height / gridSize;

  // Build patch grid scores
  const patchScores = [];
  for (let r = 0; r < gridSize; r++) {
    for (let c = 0; c < gridSize; c++) {
      const salient = sc.salientPatches.find(p => p.r === r && p.c === c);
      // Background noise is strictly low (0.05 - 0.14) so it NEVER displaces human features!
      const bgNoise = 0.06 + (((r * 11 + c * 17) % 19) / 19) * 0.08;
      const score = salient ? salient.score : bgNoise;
      const label = salient ? salient.label : null;

      patchScores.push({ r, c, score, x: c * patchW, y: r * patchH, label });
    }
  }

  // Sort and pick top-K
  const sorted = [...patchScores].sort((a, b) => b.score - a.score);
  const topKSet = new Set(sorted.slice(0, state.topK));

  // Draw semi-transparent heatmap overlay
  patchScores.forEach(p => {
    const isTopK = topKSet.has(p);

    if (isTopK) {
      ctx.fillStyle = `rgba(245, 158, 11, ${p.score * 0.65})`;
      ctx.fillRect(p.x, p.y, patchW, patchH);

      // Glowing border
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2.5;
      ctx.strokeRect(p.x + 1, p.y + 1, patchW - 2, patchH - 2);
    } else {
      ctx.fillStyle = `rgba(15, 23, 42, 0.45)`;
      ctx.fillRect(p.x, p.y, patchW, patchH);

      ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
      ctx.lineWidth = 0.5;
      ctx.strokeRect(p.x, p.y, patchW, patchH);
    }
  });

  // Calculate scores for display
  const topKMean = sorted.slice(0, state.topK).reduce((acc, p) => acc + p.score, 0) / state.topK;
  const sg = state.scenarioKey === "celebration" ? 0.682 : 0.715;
  const slocal = topKMean;
  const stot = sg + slocal;

  document.getElementById("sgScore").textContent = sg.toFixed(3);
  document.getElementById("slocalScore").textContent = slocal.toFixed(3);
  document.getElementById("stotScore").textContent = stot.toFixed(3);

  // Update prompt box
  const activeClass = state.scenarioKey === "celebration" ? "Happiness" : "Engagement";
  document.getElementById("activeClassTag").textContent = activeClass;
  const meta = EMOTIC_CLASSES.find(c => c.name === activeClass);
  if (meta) {
    document.getElementById("hardPromptText").textContent = `"${meta.hard_prompt}"`;
  }

  // Dynamic explanation under canvas
  const explanation = document.getElementById("topkExplanation");
  if (explanation) {
    if (state.scenarioKey === "celebration") {
      explanation.innerHTML = `Các ô viền sáng màu vàng là <strong>Top-${state.topK} Patches</strong> được chọn: định vị chuẩn xác vào <strong>khóe mắt nheo cười, hàm răng mở toang, 2 cánh tay vươn cao ăn mừng, cổ tay và huy chương ngực</strong>, hoàn toàn loại bỏ nền trời và ngọn cây.`;
    } else {
      explanation.innerHTML = `Các ô viền sáng màu vàng là <strong>Top-${state.topK} Patches</strong> được chọn: định vị chuẩn xác vào <strong>mắt & chân mày nhìn xuống trang sách, tay trái chống cằm, toàn bộ cánh tay phải tì bàn & bàn tay cầm bút viết, vở ghi</strong>, hoàn toàn loại bỏ giá sách và bóng đèn phía xa.`;
    }
  }

  // Patch hover tooltip listener with anatomical label
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    const c = Math.floor(mx / patchW);
    const r = Math.floor(my / patchH);

    const p = patchScores.find(item => item.r === r && item.c === c);
    if (p) {
      const isTop = topKSet.has(p);
      const tt = document.getElementById("patchTooltip");
      if (isTop && p.label) {
        tt.textContent = `Patch [${r}, ${c}] - Sim: ${p.score.toFixed(3)} ★ (Top-${state.topK}: ${p.label})`;
        tt.style.color = "#fbbf24";
      } else if (isTop) {
        tt.textContent = `Patch [${r}, ${c}] - Sim: ${p.score.toFixed(3)} ★ (Top-${state.topK} Salient Region)`;
        tt.style.color = "#fbbf24";
      } else {
        tt.textContent = `Patch [${r}, ${c}] - Sim: ${p.score.toFixed(3)} (Nhiễu nền: ${p.label || "Bối cảnh ngoài"})`;
        tt.style.color = "#94a3b8";
      }
    }
  };

  // Typeset MathJax formulas if available
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise().catch(err => {});
  }
}

// ---------- TAB 3: Fusion Diagram ----------
function renderTab3() {
  const nodeFace = document.getElementById("nodeFace");
  const nodeBody = document.getElementById("nodeBody");
  const nodeCtx  = document.getElementById("nodeContext");

  const sum = state.weights.face + state.weights.body + state.weights.context;
  const wf = state.weights.face / sum;
  const wb = state.weights.body / sum;
  const wc = state.weights.context / sum;

  // Dynamically scale nodes according to learned weights
  nodeFace.style.transform = `scale(${0.85 + wf * 0.4})`;
  nodeBody.style.transform = `scale(${0.85 + wb * 0.4})`;
  nodeCtx.style.transform  = `scale(${0.85 + wc * 0.4})`;
}

// ---------- TAB 4: Activation & Loss Dynamics ----------
function renderTab4() {
  const sc = SCENARIOS[state.scenarioKey];
  const listEl = document.getElementById("emotionBarsList");
  listEl.innerHTML = "";

  const isSoftmax = (state.activationMode === "softmax");
  const badge = document.getElementById("activationStatusBadge");
  const alertBox = document.getElementById("activationAlertBox");

  if (isSoftmax) {
    badge.className = "badge badge-danger";
    badge.textContent = "Softmax Active (LỖI ĐƠN NHÃN)";

    alertBox.className = "callout-box danger-style";
    alertBox.innerHTML = `
      <strong>BẪY THẤT BẠI KINH ĐIỂN TRÊN TẬP ĐA NHÃN:</strong><br>
      Softmax thực hiện chuẩn hóa: $p_i = \\frac{\\exp(z_i)}{\\sum_j \\exp(z_j)}$ ép buộc tổng 26 lớp $= 100\\%$.<br>
      Do <strong>${sc.gtClasses[0]}</strong> có logit cao nhất, nó chiếm tới <strong>~75%</strong>, ép toàn bộ các cảm xúc có thực khác như <em>${sc.gtClasses.slice(1).join(", ")}</em> xuống dưới 10%! Mô hình bị phạt sai và mất khả năng phát hiện đa cảm xúc cùng lúc.
    `;
  } else {
    badge.className = "badge badge-green";
    badge.textContent = "Sigmoid Active (CHUẨN ĐA NHÃN)";

    alertBox.className = "callout-box success-style";
    alertBox.innerHTML = `
      <strong>CƠ CHẾ PHÂN LOẠI ĐA NHÃN CHUẨN XÁC:</strong><br>
      Sigmoid tính độc lập từng lớp: $\\sigma(z_i) = \\frac{1}{1 + \\exp(-z_i)}$.<br>
      Nhiều cảm xúc (ví dụ: <em>${sc.gtClasses.join(" + ")}</em>) có thể đồng thời vượt ngưỡng quyết định $\\tau=0.50$, phản ánh chính xác trạng thái tâm lý phức hợp của con người trong ảnh đời thực.
    `;
  }

  // Calculate probabilities
  const logitsMap = {};
  EMOTIC_CLASSES.forEach(c => {
    logitsMap[c.name] = sc.baseLogits[c.name] !== undefined ? sc.baseLogits[c.name] : -2.0;
  });

  let probs = {};
  if (isSoftmax) {
    let expSum = 0;
    EMOTIC_CLASSES.forEach(c => {
      expSum += Math.exp(logitsMap[c.name]);
    });
    EMOTIC_CLASSES.forEach(c => {
      probs[c.name] = Math.exp(logitsMap[c.name]) / expSum;
    });
  } else {
    EMOTIC_CLASSES.forEach(c => {
      // Simulate loss effect
      let z = logitsMap[c.name];
      if (state.lossMode === "bce" && !sc.gtClasses.includes(c.name)) {
        // Unweighted BCE crushes non-GT classes drastically
        z -= 1.5;
      }
      if (state.lossMode === "bce" && sc.gtClasses.includes(c.name)) {
        // Even GT classes get suppressed by 96% negative gradient pressure
        z -= 0.8;
      }
      probs[c.name] = 1.0 / (1.0 + Math.exp(-z));
    });
  }

  // Render emotion bars
  EMOTIC_CLASSES.forEach(c => {
    const p = probs[c.name];
    const isGT = sc.gtClasses.includes(c.name);
    const isAboveThresh = p >= 0.50;

    const row = document.createElement("div");
    row.className = "emotion-row";

    const nameEl = document.createElement("div");
    nameEl.className = `emotion-name ${isGT ? "is-gt" : ""}`;
    nameEl.innerHTML = `${isGT ? "★ " : ""}${c.name}`;

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = `bar-fill ${isAboveThresh ? "bar-active" : "bar-inactive"}`;
    fill.style.width = `${Math.min(100, Math.max(0, p * 100))}%`;

    const threshLine = document.createElement("div");
    threshLine.className = "threshold-line";
    threshLine.style.left = "50%";

    track.appendChild(fill);
    track.appendChild(threshLine);

    const probText = document.createElement("div");
    probText.className = "emotion-prob";
    probText.style.color = isAboveThresh ? "#38bdf8" : "#94a3b8";
    probText.textContent = `${(p * 100).toFixed(1)}%`;

    row.appendChild(nameEl);
    row.appendChild(track);
    row.appendChild(probText);

    listEl.appendChild(row);
  });

  // Update loss gradient stats
  const gradPos = document.getElementById("gradPosVal");
  const gradNeg = document.getElementById("gradNegVal");
  const gradRatio = document.getElementById("gradRatioVal");

  if (state.lossMode === "ce") {
    gradPos.textContent = "+0.254 (Bị cạnh tranh)";
    gradNeg.textContent = "-0.746 (Quá đà)";
    gradRatio.textContent = "Sai Bản Chất";
    gradRatio.className = "l-val badge-danger";
  } else if (state.lossMode === "bce") {
    gradPos.textContent = "+0.312 (Bị áp đảo)";
    gradNeg.textContent = "-0.965 (25 lớp âm đè bẹp)";
    gradRatio.textContent = "Mất Cân Bằng (1:24)";
    gradRatio.className = "l-val badge-warning";
  } else {
    gradPos.textContent = "+0.942 (Bảo toàn)";
    gradNeg.textContent = "-0.015 (Cắt tỉa clip=0.05)";
    gradRatio.textContent = "Hài Hòa (62:1)";
    gradRatio.className = "l-val badge-green";
  }
}

// ---------- TAB 5: Metric Evaluation Lab ----------
function renderTab5() {
  renderTab5PRCurve();
  renderTab5Table();
}

function renderTab5PRCurve() {
  const canvas = document.getElementById("prCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  // Padding
  const padL = 60, padR = 30, padT = 30, padB = 50;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  // Grid & Axes
  ctx.strokeStyle = "rgba(148, 163, 184, 0.15)";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 5; i++) {
    const val = (i * 0.2).toFixed(1);
    const y = padT + plotH - (i * 0.2 * plotH);
    const x = padL + (i * 0.2 * plotW);

    // Horiz grid
    ctx.beginPath();
    ctx.moveTo(padL, y);
    ctx.lineTo(padL + plotW, y);
    ctx.stroke();

    // Vert grid
    ctx.beginPath();
    ctx.moveTo(x, padT);
    ctx.lineTo(x, padT + plotH);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#64748b";
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.textAlign = "right";
    ctx.fillText(val, padL - 8, y + 4);

    ctx.textAlign = "center";
    ctx.fillText(val, x, padT + plotH + 18);
  }

  // Axis Title
  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("Recall (Độ Phủ)", padL + plotW / 2, padT + plotH + 38);

  ctx.save();
  ctx.translate(16, padT + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Precision (Độ Chính Xác)", 0, 0);
  ctx.restore();

  // Generate synthetic PR Curve points for selected class
  const classMeta = EMOTIC_CLASSES.find(c => c.name === state.prSelectedClass) || EMOTIC_CLASSES[16];
  const targetAP = classMeta.ap / 100.0; // e.g. 0.685

  // Synthetic monotonically decreasing curve matching AP
  const points = [];
  const numSteps = 50;
  for (let s = 0; s <= numSteps; s++) {
    const r = s / numSteps;
    // Power curve model: P(r) = (1 - r^alpha) adjusted by targetAP
    const alpha = Math.max(0.2, targetAP * 2.5);
    const p = Math.max(0.05, Math.min(1.0, 1.0 - Math.pow(r, alpha) * (1.0 - targetAP * 0.5)));
    points.push({ r, p });
  }

  // Draw Area Under Curve (AP Fill)
  ctx.beginPath();
  ctx.moveTo(padL, padT + plotH);
  points.forEach(pt => {
    const px = padL + pt.r * plotW;
    const py = padT + plotH - pt.p * plotH;
    ctx.lineTo(px, py);
  });
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.closePath();
  ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
  ctx.fill();

  // Draw PR Curve Line
  ctx.beginPath();
  points.forEach((pt, idx) => {
    const px = padL + pt.r * plotW;
    const py = padT + plotH - pt.p * plotH;
    if (idx === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 3;
  ctx.stroke();

  // Operating Point on Curve at current threshold tau
  // Map tau to operating recall: tau high -> recall low, precision high
  const opRecall = Math.max(0.05, Math.min(0.95, (1.0 - state.prThreshold) * 1.1));
  const ptIndex = Math.min(numSteps, Math.floor(opRecall * numSteps));
  const opPrecision = points[ptIndex].p;

  const dotX = padL + opRecall * plotW;
  const dotY = padT + plotH - opPrecision * plotH;

  // Draw target dot
  ctx.beginPath();
  ctx.arc(dotX, dotY, 7, 0, Math.PI * 2);
  ctx.fillStyle = "#38bdf8";
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.stroke();

  // Update Stats Box
  const f1 = (2 * opPrecision * opRecall) / (opPrecision + opRecall + 1e-6);
  document.getElementById("currPrecVal").textContent = `${(opPrecision * 100).toFixed(1)}%`;
  document.getElementById("currRecVal").textContent  = `${(opRecall * 100).toFixed(1)}%`;
  document.getElementById("currF1Val").textContent   = `${(f1 * 100).toFixed(1)}%`;
  document.getElementById("currApVal").textContent   = `${(targetAP * 100).toFixed(1)}%`;
}

function renderTab5Table(filterKeyword = "") {
  const tbody = document.getElementById("apTableBody");
  tbody.innerHTML = "";

  const filtered = EMOTIC_CLASSES.filter(c => c.name.toLowerCase().includes(filterKeyword));

  filtered.forEach(c => {
    const tr = document.createElement("tr");

    const tdName = document.createElement("td");
    tdName.innerHTML = `<strong>${c.name}</strong>`;

    const tdAp = document.createElement("td");
    tdAp.className = "text-right font-mono";
    tdAp.style.color = c.ap >= 50 ? "#38bdf8" : (c.ap < 20 ? "#f43f5e" : "#e2e8f0");
    tdAp.textContent = `${c.ap.toFixed(2)}%`;

    const tdFreq = document.createElement("td");
    tdFreq.className = "text-right font-mono";
    tdFreq.textContent = c.gt_freq;

    const tdTier = document.createElement("td");
    tdTier.className = "text-center";
    if (c.tier === "top") {
      tdTier.innerHTML = `<span class="tag-top5">Top Class</span>`;
    } else if (c.tier === "tail") {
      tdTier.innerHTML = `<span class="tag-tail">Tail (Rare)</span>`;
    } else {
      tdTier.innerHTML = `<span style="color:#94a3b8; font-size:12px;">Standard</span>`;
    }

    tr.appendChild(tdName);
    tr.appendChild(tdAp);
    tr.appendChild(tdFreq);
    tr.appendChild(tdTier);

    // Row click loads into PR curve
    tr.style.cursor = "pointer";
    tr.addEventListener("click", () => {
      state.prSelectedClass = c.name;
      document.getElementById("prClassSelect").value = c.name;
      renderTab5PRCurve();
    });

    tbody.appendChild(tr);
  });
}

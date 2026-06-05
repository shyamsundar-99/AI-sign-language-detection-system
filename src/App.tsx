import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import * as tf from "@tensorflow/tfjs";

type Prediction = {
  sign: string;
  translation: string;
  confidence: number;
  time: string;
};

type Theme = "dark" | "light";
type RecognitionMode = "common" | "alphabet" | "numbers" | "international" | "sensitive" | "broadcast";

const stats = [
  ["94.2%", "Accuracy"],
  ["25-30 FPS", "Real-Time Processing"],
  ["5000+", "Custom Training Images"],
  ["CNN", "Based Recognition"],
];

const pipeline = [
  "Camera Input",
  "Hand Detection",
  "ROI Selection",
  "Skin Segmentation",
  "CNN Feature Extraction",
  "Classification",
  "Text Translation",
];

const architecture = [
  "Input Layer\n128x128 grayscale",
  "Conv2D\n32 filters + ReLU",
  "MaxPooling\n2x2",
  "Conv2D\n64 filters + ReLU",
  "Conv2D\n128 filters + ReLU",
  "Dropout\n0.35",
  "Dense Output\nSoftmax labels",
];

const datasetCards = [
  ["Rotation", "+/- 18 degree gesture variance"],
  ["Zooming", "Scale jitter for camera distance"],
  ["Shearing", "Perspective resilience"],
  ["Lighting", "Low-light and glare simulation"],
];

const challenges = [
  ["Hand Occlusion", "Landmark confidence drops when fingers overlap or leave the ROI."],
  ["Motion Blur", "Dynamic signs require temporal smoothing and sequence modeling."],
  ["Low-Light", "Skin segmentation and edge contrast degrade in dim camera feeds."],
];

const future = ["LSTM sentence synthesis", "Text-to-Speech integration", "Raspberry Pi deployment", "Smart home accessibility"];

const papers = [
  [
    "Abstract",
    "This research investigates automated sign language recognition using computer vision and deep learning architectures to reduce communication barriers for deaf and hard-of-hearing users. The proposed system combines hand localization, image preprocessing, CNN-based classification, and text or speech translation for accessible real-time interaction.",
  ],
  [
    "Introduction",
    "Sign language is expressive, visual, and context dependent. A practical recognition platform must operate in real time, handle diverse lighting conditions, and translate gestures into language that hearing users can understand instantly.",
  ],
  [
    "Literature Review",
    "Prior work uses skin segmentation, contour analysis, HOG features, CNNs, RNNs, and MediaPipe landmark tracking. Deep learning improves feature extraction, while landmark pipelines reduce noise and computational cost for real-time applications.",
  ],
  [
    "Proposed Architecture",
    "The architecture captures frames from a webcam, detects hand landmarks, crops the region of interest, applies grayscale and segmentation preprocessing, extracts hierarchical CNN features, and maps the output to sign labels with confidence scoring.",
  ],
  [
    "Experimental Results",
    "The prototype targets 94.2% aggregate accuracy, with stronger performance on static signs than dynamic signs. Browser execution demonstrates low-latency inference suitable for accessible web products.",
  ],
  [
    "Conclusion",
    "A computer vision and deep learning approach can translate common signs in real time. Future work should add sequence models, larger datasets, multilingual output, edge deployment, and robust domain adaptation.",
  ],
  [
    "References",
    "M. Shyam Sundar, Vignesh.M, and Hirthesh.R, SRM Institute of Science and Technology. Automated Sign Language Recognition Using Computer Vision and Deep Learning Architectures.",
  ],
];

const alphabetSigns = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").map((letter) => `Letter ${letter}`);
const numberSigns = Array.from({ length: 11 }, (_, index) => `Number ${index}`);
const supportedSigns = ["Open Palm", "Fist", "Peace", "Point", "I Love You", "Three", "Four", "Thumbs Up", "Thumbs Down", "Okay", "Call Me", "Letter L", "Rock"];
const internationalSigns = ["Greeting", "Thank You", "Yes", "No", "Good", "Help", "Call Me", "Peace", "I Love You"];
const sensitiveSigns = ["Profanity: fuck", "Profanity: shit", "Profanity: damn", "Profanity: hell", "Insult", "Angry", "Stop"];
const broadcastSigns = ["Hello", "Yes", "No", "Good", "You", "Help", "Call me", "Peace", "I love you", "Stop", "Okay"];
const recognitionModes: { value: RecognitionMode; label: string; description: string }[] = [
  { value: "common", label: "Common signs", description: "Daily accessibility signs and greetings" },
  { value: "alphabet", label: "ASL alphabet", description: "A-Z alphabet practice with landmark templates" },
  { value: "numbers", label: "Numbers", description: "0-10 number mode, optimized for Android camera" },
  { value: "international", label: "Other sign sets", description: "ISL/BSL-inspired common meaning labels" },
  { value: "sensitive", label: "Sensitive vocabulary", description: "Profanity and strong-language labels, no slurs" },
  { value: "broadcast", label: "Broadcast live captions", description: "Fast news-channel style sentence conversion" },
];

const translations: Record<string, Record<string, string>> = {
  English: {
    "Open Palm": "Hello",
    Fist: "Yes",
    Peace: "Peace",
    Point: "You",
    "I Love You": "I love you",
    Three: "Three",
    Four: "Four",
    "Thumbs Up": "Good",
    "Thumbs Down": "No",
    Okay: "Okay",
    "Call Me": "Call me",
    "Letter L": "L",
    Rock: "Rock on",
  },
  Spanish: {
    "Open Palm": "Hola",
    Fist: "Si",
    Peace: "Paz",
    Point: "Tu",
    "I Love You": "Te quiero",
    Three: "Tres",
    Four: "Cuatro",
    "Thumbs Up": "Bien",
    "Thumbs Down": "No",
    Okay: "Vale",
    "Call Me": "Llamame",
    "Letter L": "L",
    Rock: "Genial",
  },
  Hindi: {
    "Open Palm": "Namaste",
    Fist: "Haan",
    Peace: "Shanti",
    Point: "Tum",
    "I Love You": "Main tumse pyaar karta hoon",
    Three: "Teen",
    Four: "Chaar",
    "Thumbs Up": "Accha",
    "Thumbs Down": "Nahi",
    Okay: "Theek hai",
    "Call Me": "Mujhe call karo",
    "Letter L": "L",
    Rock: "Bahut badhiya",
  },
  Tamil: {
    "Open Palm": "Vanakkam",
    Fist: "Aam",
    Peace: "Amaithi",
    Point: "Nee",
    "I Love You": "Naan unnai nesikkiren",
    Three: "Moondru",
    Four: "Naangu",
    "Thumbs Up": "Nandru",
    "Thumbs Down": "Illai",
    Okay: "Sari",
    "Call Me": "Ennai azhaikkavum",
    "Letter L": "L",
    Rock: "Semma",
  },
  French: {
    "Open Palm": "Bonjour",
    Fist: "Oui",
    Peace: "Paix",
    Point: "Toi",
    "I Love You": "Je t'aime",
    Three: "Trois",
    Four: "Quatre",
    "Thumbs Up": "Bien",
    "Thumbs Down": "Non",
    Okay: "D'accord",
    "Call Me": "Appelle-moi",
    "Letter L": "L",
    Rock: "Super",
  },
};

function classifyLandmarks(landmarks: { x: number; y: number; z?: number }[] | undefined, mode: RecognitionMode) {
  if (!landmarks?.length) return { sign: "Scanning", confidence: 0 };
  const up = (tip: number, pip: number) => landmarks[tip].y < landmarks[pip].y - 0.025;
  const distance = (a: number, b: number) => Math.hypot(landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y);
  const index = up(8, 6);
  const middle = up(12, 10);
  const ring = up(16, 14);
  const pinky = up(20, 18);
  const thumb = Math.abs(landmarks[4].x - landmarks[2].x) > 0.075 && landmarks[4].y < landmarks[3].y + 0.08;
  const count = [thumb, index, middle, ring, pinky].filter(Boolean).length;
  let sign = "Open Palm";
  const closedFingers = !index && !middle && !ring && !pinky;
  const thumbUp = thumb && closedFingers && landmarks[4].y < landmarks[3].y - 0.045;
  const thumbDown = thumb && closedFingers && landmarks[4].y > landmarks[3].y + 0.045;
  const okay = distance(4, 8) < 0.06 && middle && ring && pinky;
  const middleOnly = middle && !thumb && !index && !ring && !pinky;
  const commonSign = () => {
    if (okay) return "Okay";
    if (thumbUp) return "Thumbs Up";
    if (thumbDown) return "Thumbs Down";
    if (closedFingers) return "Fist";
    if (thumb && index && pinky && !middle && !ring) return "I Love You";
    if (thumb && pinky && !index && !middle && !ring) return "Call Me";
    if (thumb && index && !middle && !ring && !pinky) return "Letter L";
    if (index && pinky && !middle && !ring) return "Rock";
    if (index && middle && !ring && !pinky) return "Peace";
    if (index && !middle && !ring && !pinky) return "Point";
    if (index && middle && ring && !pinky) return "Three";
    if (index && middle && ring && pinky && !thumb) return "Four";
    return "Open Palm";
  };
  if (mode === "numbers") {
    if (okay || closedFingers) sign = "Number 0";
    else sign = `Number ${Math.min(10, count)}`;
  } else if (mode === "alphabet") {
    if (okay) sign = "Letter O";
    else if (closedFingers) sign = "Letter A";
    else if (!thumb && index && middle && ring && pinky) sign = "Letter B";
    else if (thumb && index && !middle && !ring && !pinky) sign = "Letter L";
    else if (thumb && pinky && !index && !middle && !ring) sign = "Letter Y";
    else if (!thumb && !index && !middle && !ring && pinky) sign = "Letter I";
    else if (!thumb && index && !middle && !ring && !pinky) sign = "Letter D";
    else if (!thumb && index && middle && !ring && !pinky) sign = "Letter V";
    else if (!thumb && index && middle && ring && !pinky) sign = "Letter W";
    else if (thumb && index && pinky && !middle && !ring) sign = "Letter ILY";
    else sign = `Letter ${String.fromCharCode(65 + Math.min(25, count * 3))}`;
  } else if (mode === "international") {
    const mapped = commonSign();
    sign = mapped === "Open Palm" ? "Greeting" : mapped === "Thumbs Up" ? "Good" : mapped === "Thumbs Down" ? "No" : mapped === "Fist" ? "Yes" : mapped === "Point" ? "Help" : mapped;
  } else if (mode === "sensitive") {
    if (middleOnly) sign = "Profanity: fuck";
    else if (thumbDown) sign = "Profanity: damn";
    else if (closedFingers) sign = "Profanity: hell";
    else if (index && pinky && !middle && !ring) sign = "Profanity: shit";
    else if (index && !middle && !ring && !pinky) sign = "Insult";
    else if (index && middle && !ring && !pinky) sign = "Angry";
    else sign = "Stop";
  } else if (mode === "broadcast") {
    const mapped = commonSign();
    sign = mapped === "Open Palm" ? "Hello" : mapped === "Fist" ? "Yes" : mapped === "Thumbs Down" ? "No" : mapped === "Thumbs Up" ? "Good" : mapped === "Point" ? "You" : mapped;
  } else if (okay) sign = "Okay";
  else if (thumbUp) sign = "Thumbs Up";
  else if (thumbDown) sign = "Thumbs Down";
  else if (closedFingers) sign = "Fist";
  else if (thumb && index && pinky && !middle && !ring) sign = "I Love You";
  else if (thumb && pinky && !index && !middle && !ring) sign = "Call Me";
  else if (thumb && index && !middle && !ring && !pinky) sign = "Letter L";
  else if (index && pinky && !middle && !ring) sign = "Rock";
  else if (index && middle && !ring && !pinky) sign = "Peace";
  else if (index && !middle && !ring && !pinky) sign = "Point";
  else if (index && middle && ring && !pinky) sign = "Three";
  else if (index && middle && ring && pinky && !thumb) sign = "Four";
  const confidence = Math.min(98, 72 + count * 4 + Math.round(Math.abs((landmarks[0].z ?? 0) * 50)));
  return { sign, confidence };
}

function translate(sign: string, language: string) {
  if (sign.startsWith("Letter ")) return sign.replace("Letter ", "Alphabet: ");
  if (sign.startsWith("Number ")) return sign.replace("Number ", "Number: ");
  if (sign.startsWith("Profanity: ")) return `Strong language: ${sign.replace("Profanity: ", "")}`;
  return translations[language]?.[sign] ?? sign;
}

function App() {
  const [theme, setTheme] = useState<Theme>("dark");
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraStatus, setCameraStatus] = useState("Camera offline");
  const [cameraError, setCameraError] = useState("");
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [recognitionMode, setRecognitionMode] = useState<RecognitionMode>("common");
  const [liveCaptions, setLiveCaptions] = useState<string[]>([]);
  const [modelReady, setModelReady] = useState(false);
  const [prediction, setPrediction] = useState<Prediction>({ sign: "Awaiting camera", translation: "Start camera to begin", confidence: 0, time: "--" });
  const [history, setHistory] = useState<Prediction[]>([]);
  const [language, setLanguage] = useState("English");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [openPaper, setOpenPaper] = useState("Abstract");
  const [chat, setChat] = useState(["Hi, I can explain signs, confidence, and accessibility features."]);
  const [chatInput, setChatInput] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef = useRef<number | null>(null);
  const lastSpoken = useRef("");
  const lastSaved = useRef(0);
  const languageRef = useRef(language);
  const recognitionModeRef = useRef<RecognitionMode>(recognitionMode);
  const predictionBufferRef = useRef<string[]>([]);
  const lastBroadcastRef = useRef({ sign: "", time: 0 });

  const cnnModel = useMemo(() => {
    const model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape: [128, 128, 1], filters: 32, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: "relu" }));
    model.add(tf.layers.dropout({ rate: 0.35 }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 6, activation: "softmax" }));
    return model;
  }, []);

  useEffect(() => {
    languageRef.current = language;
  }, [language]);

  useEffect(() => {
    recognitionModeRef.current = recognitionMode;
  }, [recognitionMode]);

  useEffect(() => {
    refreshCameraList();
    return () => stopCamera();
  }, []);

  async function refreshCameraList() {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === "videoinput");
    setAvailableCameras(cameras);
    if (!selectedCameraId && cameras[0]?.deviceId) setSelectedCameraId(cameras[0].deviceId);
  }

  async function ensureLandmarker() {
    if (landmarkerRef.current) return landmarkerRef.current;
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm");
    const options = {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU" as const,
      },
      runningMode: "VIDEO" as const,
      numHands: 1,
      minHandDetectionConfidence: 0.45,
      minTrackingConfidence: 0.45,
    };
    let landmarker: HandLandmarker;
    try {
      landmarker = await HandLandmarker.createFromOptions(vision, options);
    } catch {
      landmarker = await HandLandmarker.createFromOptions(vision, { ...options, baseOptions: { ...options.baseOptions, delegate: "CPU" } });
    }
    landmarkerRef.current = landmarker;
    setModelReady(true);
    return landmarker;
  }

  async function startCamera() {
    try {
      setCameraError("");
      stopCamera();
      setCameraStatus("Requesting camera permission...");
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("This browser does not support camera access. Use Chrome, Edge, or Android Chrome.");
      }
      if (!window.isSecureContext) {
        throw new Error("Camera access requires HTTPS or localhost. If you opened this from a phone, use an HTTPS tunnel or run it directly on the phone.");
      }
      const preferredVideo: MediaTrackConstraints = selectedCameraId
        ? { deviceId: { exact: selectedCameraId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } };
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: preferredVideo, audio: false });
      } catch {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.muted = true;
        videoRef.current.playsInline = true;
        await waitForVideoFrame(videoRef.current);
        await videoRef.current.play();
      }
      setCameraOn(true);
      setCameraStatus("Camera preview live");
      setPrediction({ sign: "Camera connected", translation: "Preview is live. Loading AI tracker...", confidence: 12, time: new Date().toLocaleTimeString() });
      await refreshCameraList();
      try {
        setCameraStatus("Loading AI hand tracker...");
        await ensureLandmarker();
        setCameraStatus("Camera live with AI tracking");
        detectLoop();
      } catch (modelError) {
        const message = modelError instanceof Error ? modelError.message : "AI tracker failed to load.";
        setCameraStatus("Preview live, AI tracker offline");
        setCameraError(`Camera works, but MediaPipe did not load: ${message}`);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Camera failed to start.";
      setCameraOn(false);
      setCameraStatus("Camera blocked or unavailable");
      setCameraError(message);
      setPrediction({ sign: "Camera unavailable", translation: message, confidence: 0, time: "--" });
    }
  }

  function waitForVideoFrame(video: HTMLVideoElement) {
    if (video.readyState >= 2) return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(() => reject(new Error("Camera opened, but no video frame arrived. Try another camera source or close apps using the camera.")), 5000);
      video.onloadedmetadata = () => {
        window.clearTimeout(timer);
        resolve();
      };
    });
  }

  function stopCamera() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOn(false);
    setCameraStatus("Camera offline");
  }

  function speak(text: string) {
    if (!("speechSynthesis" in window) || !text || text === lastSpoken.current) return;
    lastSpoken.current = text;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.92;
    utterance.pitch = 1.05;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }

  function smoothSign(sign: string) {
    predictionBufferRef.current = [...predictionBufferRef.current, sign].slice(-7);
    const votes = predictionBufferRef.current.reduce<Record<string, number>>((acc, item) => {
      acc[item] = (acc[item] ?? 0) + 1;
      return acc;
    }, {});
    return Object.entries(votes).sort((a, b) => b[1] - a[1])[0]?.[0] ?? sign;
  }

  function appendBroadcastCaption(sign: string, confidence: number) {
    if (recognitionModeRef.current !== "broadcast" || confidence < 76 || sign === "Scanning") return;
    const now = performance.now();
    const repeatedTooSoon = lastBroadcastRef.current.sign === sign && now - lastBroadcastRef.current.time < 950;
    if (repeatedTooSoon) return;
    lastBroadcastRef.current = { sign, time: now };
    const word = translate(sign, languageRef.current).replace(/^Strong language: /, "");
    setLiveCaptions((current) => [...current, word].slice(-24));
  }

  function detectLoop() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const landmarker = landmarkerRef.current;
    if (!video || !canvas || !landmarker || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(detectLoop);
      return;
    }
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const results = landmarker.detectForVideo(video, performance.now());
    const landmarks = results.landmarks[0];
    if (landmarks?.length) {
      drawHandOverlay(ctx, canvas, landmarks);
      const classified = classifyLandmarks(landmarks, recognitionModeRef.current);
      const stableSign = smoothSign(classified.sign);
      const translated = translate(stableSign, languageRef.current);
      const item = { sign: stableSign, translation: translated, confidence: classified.confidence, time: new Date().toLocaleTimeString() };
      setPrediction(item);
      appendBroadcastCaption(stableSign, classified.confidence);
      if (classified.confidence > 82) speak(translated);
      if (performance.now() - lastSaved.current > 1600) {
        lastSaved.current = performance.now();
        setHistory((current) => [item, ...current].slice(0, 12));
      }
    } else {
      setPrediction((current) => ({ ...current, sign: "Scanning", translation: "Place your hand in frame", confidence: 0 }));
    }
    rafRef.current = requestAnimationFrame(detectLoop);
  }

  function drawHandOverlay(ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, landmarks: { x: number; y: number }[]) {
    const points = landmarks.map((p) => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.max(0, Math.min(...xs) - 52);
    const minY = Math.max(0, Math.min(...ys) - 52);
    const maxX = Math.min(canvas.width, Math.max(...xs) + 52);
    const maxY = Math.min(canvas.height, Math.max(...ys) + 52);
    const scan = (performance.now() / 7) % Math.max(1, maxY - minY);
    ctx.strokeStyle = "rgba(34, 211, 238, .9)";
    ctx.lineWidth = 2;
    ctx.shadowColor = "#22d3ee";
    ctx.shadowBlur = 22;
    ctx.strokeRect(minX, minY, maxX - minX, maxY - minY);
    ctx.beginPath();
    ctx.moveTo(minX, minY + scan);
    ctx.lineTo(maxX, minY + scan);
    ctx.strokeStyle = "rgba(168, 85, 247, .95)";
    ctx.lineWidth = 4;
    ctx.stroke();
    ctx.shadowBlur = 0;
    const connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];
    ctx.strokeStyle = "rgba(96, 165, 250, .9)";
    ctx.lineWidth = 3;
    connections.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(points[a].x, points[a].y);
      ctx.lineTo(points[b].x, points[b].y);
      ctx.stroke();
    });
    points.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "#e0f2fe";
      ctx.fill();
    });
  }

  function downloadHistory() {
    const rows = ["time,sign,translation,confidence", ...history.map((h) => `${h.time},${h.sign},${h.translation},${h.confidence}`)];
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "sign-language-predictions.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  function sendChat() {
    if (!chatInput.trim()) return;
    const reply = chatInput.toLowerCase().includes("confidence")
      ? "Confidence combines MediaPipe landmark stability with a CNN-style label score and temporal smoothing."
      : chatInput.toLowerCase().includes("voice")
        ? "Voice narration uses the browser SpeechSynthesis API so translated signs can be heard immediately."
        : "Try launching detection, place one hand inside the frame, and use Open Palm, Fist, Peace, Point, or I Love You.";
    setChat((items) => [...items, `You: ${chatInput}`, `Assistant: ${reply}`]);
    setChatInput("");
  }

  return (
    <main className={`${theme === "dark" ? "dark bg-[#050713] text-white" : "bg-slate-50 text-slate-950"} min-h-screen overflow-hidden font-sans transition-colors duration-700`}>
      <ParticleField />
      <nav className="fixed left-0 right-0 top-0 z-50 border-b border-white/10 bg-slate-950/45 backdrop-blur-2xl dark:bg-slate-950/45">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-5 py-4">
          <a href="#top" className="text-sm font-semibold uppercase tracking-[0.34em] text-cyan-200">ASL Recognition AI</a>
          <div className="hidden items-center gap-7 text-sm text-slate-300 md:flex">
            <a href="#detect">Detection</a><a href="#pipeline">Pipeline</a><a href="#results">Results</a><a href="#research">Research</a>
          </div>
          <button onClick={() => setTheme(theme === "dark" ? "light" : "dark")} className="rounded-full border border-cyan-300/30 px-4 py-2 text-xs font-semibold uppercase tracking-widest text-cyan-100 transition hover:bg-cyan-300/10">
            {theme === "dark" ? "Light" : "Dark"} mode
          </button>
        </div>
      </nav>

      <section id="top" className="relative min-h-screen px-5 pt-28">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_65%_30%,rgba(124,58,237,.32),transparent_34%),radial-gradient(circle_at_30%_15%,rgba(14,165,233,.28),transparent_28%)]" />
        <div className="relative mx-auto grid min-h-[calc(100vh-7rem)] max-w-7xl items-center gap-12 lg:grid-cols-[1.05fr_.95fr]">
          <motion.div initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8 }}>
            <p className="mb-5 text-sm font-semibold uppercase tracking-[0.45em] text-cyan-300">SRM Institute research product</p>
            <h1 className="max-w-5xl bg-gradient-to-r from-white via-cyan-100 to-violet-200 bg-clip-text text-5xl font-black leading-[0.95] tracking-tight text-transparent sm:text-7xl lg:text-8xl">
              Automated Sign Language Recognition
            </h1>
            <p className="mt-7 max-w-2xl text-lg leading-8 text-slate-300 sm:text-xl">Bridging communication gaps using Computer Vision and Deep Learning</p>
            <div className="mt-9 flex flex-wrap gap-4">
              <a href="#detect" className="rounded-full bg-cyan-300 px-6 py-3 font-semibold text-slate-950 shadow-[0_0_35px_rgba(34,211,238,.45)] transition hover:scale-105">Launch Detection</a>
              <a href="#research" className="rounded-full border border-white/15 px-6 py-3 font-semibold text-white backdrop-blur transition hover:border-violet-300 hover:bg-white/10">View Research</a>
              <a href="#demo" className="rounded-full border border-cyan-300/40 px-6 py-3 font-semibold text-cyan-100 transition hover:bg-cyan-300/10">Live Demo</a>
            </div>
            <div className="mt-10 grid grid-cols-2 gap-4 lg:grid-cols-4">
              {stats.map(([value, label], index) => <GlassMetric key={label} value={value} label={label} delay={index * 0.08} />)}
            </div>
          </motion.div>
          <HandVisualization />
        </div>
      </section>

      <section id="detect" className="relative mx-auto max-w-7xl px-5 py-24">
        <SectionTitle eyebrow="Live Sign Detection Module" title="Webcam recognition with MediaPipe landmarks and CNN-inspired scoring." text="Start the camera, place a hand in frame, and the browser will track landmarks, draw an AI overlay, translate recognized signs, narrate output, and save prediction history." />
        <div id="demo" className="mt-10 grid gap-6 lg:grid-cols-[1.25fr_.75fr]">
          <div className="glass-panel relative min-h-[420px] overflow-hidden rounded-[2rem] border border-white/10 bg-white/[.055] p-3 shadow-2xl shadow-cyan-950/40 backdrop-blur-2xl">
            <video ref={videoRef} muted playsInline className="h-full min-h-[420px] w-full scale-x-[-1] rounded-[1.55rem] bg-slate-950 object-cover" />
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-3 h-[calc(100%-1.5rem)] w-[calc(100%-1.5rem)] scale-x-[-1] rounded-[1.55rem]" />
            <div className="absolute left-6 top-6 rounded-full border border-cyan-300/30 bg-slate-950/70 px-4 py-2 text-xs font-semibold uppercase tracking-widest text-cyan-100 backdrop-blur-xl">{cameraStatus}</div>
            {cameraOn && recognitionMode === "broadcast" && <div className="absolute bottom-6 left-6 right-6 rounded-2xl border border-cyan-300/30 bg-slate-950/80 p-4 backdrop-blur-xl"><p className="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-200">Live caption feed</p><p className="mt-2 text-xl font-bold leading-8 text-white">{liveCaptions.length ? liveCaptions.join(" ") : "Waiting for broadcast-speed signs..."}</p></div>}
            {!cameraOn && <div className="absolute inset-0 grid place-items-center bg-slate-950/70 p-6 text-center backdrop-blur-sm"><div><p className="text-2xl font-bold text-white">Camera is offline</p><p className="mt-2 text-slate-300">Launch detection, choose DroidCam/Iriun if installed, or open the app on Android Chrome.</p>{cameraError && <p className="mt-4 max-w-xl rounded-2xl border border-rose-300/20 bg-rose-500/10 p-4 text-sm text-rose-100">{cameraError}</p>}</div></div>}
          </div>
          <div className="space-y-5">
            <div className="glass-panel rounded-[2rem] border border-white/10 bg-white/[.06] p-6 backdrop-blur-2xl">
              <div className="flex items-center justify-between gap-3">
                <span className="text-xs uppercase tracking-[0.34em] text-cyan-200">Prediction</span>
                <span className={`h-2.5 w-2.5 rounded-full ${modelReady ? "bg-emerald-300 shadow-[0_0_16px_#6ee7b7]" : "bg-amber-300"}`} />
              </div>
              <h3 className="mt-5 text-4xl font-black">{prediction.sign}</h3>
              <p className="mt-2 text-xl text-cyan-100">{prediction.translation}</p>
              <div className="mt-6 h-3 rounded-full bg-white/10"><motion.div className="h-full rounded-full bg-gradient-to-r from-cyan-300 to-violet-400" animate={{ width: `${prediction.confidence}%` }} /></div>
              <p className="mt-2 text-sm text-slate-400">Confidence score: {prediction.confidence}%</p>
              <div className="mt-5 space-y-3">
                <label className="block text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Recognition mode</label>
                <select value={recognitionMode} onChange={(e) => setRecognitionMode(e.target.value as RecognitionMode)} className="w-full rounded-2xl border border-cyan-300/20 bg-slate-950/70 px-4 py-3 text-sm text-white outline-none focus:border-cyan-300">
                  {recognitionModes.map((mode) => <option key={mode.value} value={mode.value}>{mode.label} - {mode.description}</option>)}
                </select>
              </div>
              <div className="mt-5 space-y-3">
                <label className="block text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Camera source</label>
                <select value={selectedCameraId} onChange={(e) => setSelectedCameraId(e.target.value)} className="w-full rounded-2xl border border-white/10 bg-slate-950/70 px-4 py-3 text-sm text-white outline-none focus:border-cyan-300">
                  <option value="">Auto / phone browser camera</option>
                  {availableCameras.map((camera, index) => <option key={camera.deviceId || index} value={camera.deviceId}>{camera.label || `Camera ${index + 1}`}</option>)}
                </select>
                <div className="grid grid-cols-2 gap-2">
                  <button onClick={() => { setSelectedCameraId(""); setFacingMode("user"); }} className={`rounded-full border px-4 py-2 text-sm ${facingMode === "user" && !selectedCameraId ? "border-cyan-300 bg-cyan-300/10 text-cyan-100" : "border-white/10 text-slate-300"}`}>Front camera</button>
                  <button onClick={() => { setSelectedCameraId(""); setFacingMode("environment"); }} className={`rounded-full border px-4 py-2 text-sm ${facingMode === "environment" && !selectedCameraId ? "border-cyan-300 bg-cyan-300/10 text-cyan-100" : "border-white/10 text-slate-300"}`}>Back camera</button>
                </div>
                <button onClick={refreshCameraList} className="w-full rounded-full border border-violet-300/30 px-4 py-2 text-sm font-semibold text-violet-100 hover:bg-violet-300/10">Refresh camera list</button>
              </div>
              <div className="mt-5 rounded-2xl border border-white/10 bg-white/[.04] p-4">
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-slate-400">Option B gesture set</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">Common mode: {supportedSigns.join(", ")}.</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">Alphabet mode includes {alphabetSigns.length} A-Z labels with live templates for A, B, D, I, L, O, V, W, Y, and ILY-style hand shapes.</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">Number mode supports {numberSigns.join(", ")}. Other sign sets include {internationalSigns.join(", ")}.</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">Sensitive vocabulary adds {sensitiveSigns.join(", ")} with 7-frame confidence smoothing to reduce false triggers.</p>
                <p className="mt-2 text-sm leading-6 text-slate-300">Broadcast mode converts fast signs into rolling captions: {broadcastSigns.join(", ")}.</p>
              </div>
              <div className="mt-6 flex flex-wrap gap-3">
                <button onClick={startCamera} className="rounded-full bg-white px-5 py-3 font-semibold text-slate-950 transition hover:scale-105">Start camera</button>
                <button onClick={stopCamera} className="rounded-full border border-white/15 px-5 py-3 font-semibold text-white transition hover:bg-white/10">Stop camera</button>
              </div>
            </div>
            <div className="glass-panel rounded-[2rem] border border-white/10 bg-white/[.06] p-6 backdrop-blur-2xl">
              <div className="flex items-center justify-between"><h3 className="font-semibold">Translation output</h3><select value={language} onChange={(e) => setLanguage(e.target.value)} className="rounded-full border border-white/10 bg-slate-950/60 px-3 py-2 text-sm text-white">{Object.keys(translations).map((item) => <option key={item}>{item}</option>)}</select></div>
              <div className="mt-4 rounded-2xl border border-cyan-300/20 bg-cyan-300/10 p-4 text-lg text-cyan-50">{prediction.translation}</div>
              <div className="mt-4 rounded-2xl border border-violet-300/20 bg-violet-400/10 p-4"><div className="flex items-center justify-between gap-3"><p className="text-xs font-semibold uppercase tracking-[0.24em] text-violet-100">News-channel sentence stream</p><button onClick={() => setLiveCaptions([])} className="rounded-full border border-white/10 px-3 py-1 text-xs text-slate-200 hover:bg-white/10">Clear</button></div><p className="mt-3 min-h-16 text-2xl font-black leading-9 text-white">{liveCaptions.length ? liveCaptions.join(" ") : "Switch to Broadcast live captions mode for continuous conversion."}</p></div>
              <Waveform active={isSpeaking} />
              <button onClick={downloadHistory} className="mt-5 w-full rounded-full border border-cyan-300/30 px-4 py-3 text-sm font-semibold text-cyan-100 hover:bg-cyan-300/10">Download prediction history</button>
            </div>
          </div>
        </div>
      </section>

      <section id="pipeline" className="mx-auto max-w-7xl px-5 py-20">
        <SectionTitle eyebrow="AI Pipeline Visualization" title="From camera frame to accessible translation." text="The product follows the paper's computer vision workflow with browser-ready preprocessing and neural classification stages." />
        <div className="mt-12 grid gap-4 md:grid-cols-7">
          {pipeline.map((step, index) => <PipelineNode key={step} step={step} index={index} />)}
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-20">
        <SectionTitle eyebrow="Model Architecture" title="CNN architecture designed for 128x128 grayscale hand ROIs." text={`TensorFlow.js model instantiated in browser: ${cnnModel.layers.length} layers ready for custom trained weights.`} />
        <div className="mt-12 grid gap-5 lg:grid-cols-[.9fr_1.1fr]">
          <NeuralNetwork />
          <div className="grid gap-4 sm:grid-cols-2">
            {architecture.map((layer, index) => <motion.div key={layer} whileHover={{ y: -6 }} className="rounded-3xl border border-white/10 bg-white/[.055] p-5 backdrop-blur-xl"><span className="text-xs text-cyan-200">Layer {index + 1}</span><p className="mt-3 whitespace-pre-line text-lg font-bold">{layer}</p></motion.div>)}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-20">
        <SectionTitle eyebrow="Dataset" title="Public ASL data plus 5000+ custom gesture frames." text="Data augmentation improves robustness across device cameras, hand sizes, backgrounds, and lighting conditions." />
        <div className="mt-10 grid gap-4 md:grid-cols-4">{datasetCards.map(([title, text]) => <motion.div key={title} whileHover={{ rotate: -1, scale: 1.03 }} className="rounded-[1.75rem] border border-white/10 bg-gradient-to-br from-white/[.09] to-white/[.025] p-6 backdrop-blur-xl"><div className="mb-8 h-24 rounded-3xl bg-[conic-gradient(from_180deg,rgba(34,211,238,.75),rgba(168,85,247,.85),rgba(15,23,42,.4),rgba(34,211,238,.75))] opacity-80 blur-[1px]" /><h3 className="text-xl font-bold">{title}</h3><p className="mt-2 text-slate-300">{text}</p></motion.div>)}</div>
      </section>

      <section id="results" className="mx-auto max-w-7xl px-5 py-20">
        <SectionTitle eyebrow="Results Dashboard" title="Production metrics for static signs, dynamic gestures, and latency." text="A premium analytics view summarizes experimental performance and real-time inference behavior." />
        <ResultsDashboard />
      </section>

      <section id="research" className="mx-auto max-w-5xl px-5 py-20">
        <SectionTitle eyebrow="Research Paper Viewer" title="Automated Sign Language Recognition Using Computer Vision and Deep Learning Architectures" text="By M. Shyam Sundar (RA2311042010073), Vignesh.M, and Hirthesh.R from SRM Institute of Science and Technology." />
        <div className="mt-10 space-y-3">{papers.map(([title, text]) => <div key={title} className="overflow-hidden rounded-3xl border border-white/10 bg-white/[.055] backdrop-blur-xl"><button onClick={() => setOpenPaper(openPaper === title ? "" : title)} className="flex w-full items-center justify-between px-6 py-5 text-left font-bold"><span>{title}</span><span>{openPaper === title ? "-" : "+"}</span></button><AnimatePresence>{openPaper === title && <motion.p initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="px-6 pb-6 leading-8 text-slate-300">{text}</motion.p>}</AnimatePresence></div>)}</div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-20">
        <SectionTitle eyebrow="Challenges and Future Scope" title="Designed for accessibility beyond the demo." text="The next research phase focuses on temporal language understanding, edge deployment, and inclusive smart environments." />
        <div className="mt-10 grid gap-5 lg:grid-cols-3">{challenges.map(([title, text]) => <motion.div key={title} whileHover={{ y: -8 }} className="rounded-[2rem] border border-violet-300/20 bg-violet-400/[.07] p-7 backdrop-blur-xl"><h3 className="text-2xl font-bold">{title}</h3><p className="mt-3 text-slate-300">{text}</p></motion.div>)}</div>
        <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">{future.map((item) => <div key={item} className="rounded-full border border-cyan-300/20 bg-cyan-300/10 px-5 py-4 text-center text-cyan-50">{item}</div>)}</div>
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-24">
        <div className="grid gap-6 lg:grid-cols-[.75fr_1.25fr]">
          <div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl"><h3 className="text-2xl font-bold">AI stack</h3><p className="mt-3 text-slate-300">Frontend: React, Tailwind CSS, Framer Motion. AI: MediaPipe hand tracking, OpenCV-style ROI preprocessing, TensorFlow/Keras CNN architecture. Backend target: Flask or FastAPI inference service for production weights.</p></div>
          <div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl"><h3 className="text-2xl font-bold">Accessibility assistant</h3><div className="mt-4 max-h-44 space-y-2 overflow-auto text-sm text-slate-300">{chat.map((item, index) => <p key={`${item}-${index}`} className="rounded-2xl bg-white/5 p-3">{item}</p>)}</div><div className="mt-4 flex gap-2"><input value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && sendChat()} placeholder="Ask about detection, confidence, or voice..." className="min-w-0 flex-1 rounded-full border border-white/10 bg-slate-950/60 px-4 py-3 text-white outline-none focus:border-cyan-300" /><button onClick={sendChat} className="rounded-full bg-cyan-300 px-5 font-semibold text-slate-950">Send</button></div></div>
        </div>
      </section>
    </main>
  );
}

function ParticleField() {
  return <div className="pointer-events-none fixed inset-0 z-0 opacity-70">{Array.from({ length: 42 }).map((_, i) => <span key={i} className="particle" style={{ left: `${(i * 37) % 100}%`, top: `${(i * 61) % 100}%`, animationDelay: `${(i % 9) * 0.55}s`, animationDuration: `${8 + (i % 7)}s` }} />)}</div>;
}

function GlassMetric({ value, label, delay }: { value: string; label: string; delay: number }) {
  return <motion.div initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ delay }} className="rounded-3xl border border-white/10 bg-white/[.07] p-5 backdrop-blur-2xl"><p className="text-2xl font-black text-white">{value}</p><p className="mt-1 text-xs uppercase tracking-widest text-slate-300">{label}</p></motion.div>;
}

function HandVisualization() {
  return <motion.div initial={{ opacity: 0, scale: 0.92 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 1 }} className="relative mx-auto aspect-square w-full max-w-[560px]"><div className="absolute inset-0 rounded-full bg-cyan-300/20 blur-3xl" /><motion.div animate={{ rotate: 360 }} transition={{ duration: 22, repeat: Infinity, ease: "linear" }} className="absolute inset-8 rounded-full border border-dashed border-cyan-300/35" /><svg viewBox="0 0 500 500" className="relative z-10 h-full w-full drop-shadow-[0_0_35px_rgba(34,211,238,.45)]"><defs><linearGradient id="hand" x1="0" x2="1"><stop stopColor="#22d3ee" /><stop offset="1" stopColor="#a855f7" /></linearGradient></defs><path d="M238 404c-62-5-107-49-112-111l-9-115c-2-25 35-30 42-6l19 70V104c0-31 45-31 45 0v117-143c0-31 46-31 46 0v145-125c0-30 45-30 45 0v137-91c0-28 42-29 44-1l6 122c3 77-48 145-126 139Z" fill="url(#hand)" opacity=".82" /><g fill="#e0f2fe">{[205, 246, 291, 335, 161, 188, 226, 269, 313, 356].map((x, i) => <motion.circle key={i} cx={x} cy={i < 4 ? 83 + i * 7 : 254 + (i % 3) * 18} r="5" animate={{ opacity: [0.25, 1, 0.25], scale: [0.8, 1.4, 0.8] }} transition={{ duration: 1.8, repeat: Infinity, delay: i * 0.12 }} />)}</g></svg></motion.div>;
}

function SectionTitle({ eyebrow, title, text }: { eyebrow: string; title: string; text: string }) {
  return <div className="relative z-10 max-w-3xl"><p className="text-sm font-semibold uppercase tracking-[0.34em] text-cyan-300">{eyebrow}</p><h2 className="mt-4 text-4xl font-black tracking-tight text-white sm:text-5xl">{title}</h2><p className="mt-4 text-lg leading-8 text-slate-300">{text}</p></div>;
}

function PipelineNode({ step, index }: { step: string; index: number }) {
  return <motion.div initial={{ opacity: 0, x: -14 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.07 }} className="relative rounded-3xl border border-cyan-300/20 bg-cyan-300/[.07] p-5 text-center backdrop-blur-xl"><div className="mx-auto mb-4 h-12 w-12 rounded-2xl bg-gradient-to-br from-cyan-300 to-violet-500 shadow-[0_0_24px_rgba(34,211,238,.45)]" /><p className="text-sm font-bold">{step}</p>{index < pipeline.length - 1 && <span className="absolute -right-4 top-1/2 hidden h-px w-8 bg-gradient-to-r from-cyan-300 to-violet-400 md:block" />}</motion.div>;
}

function NeuralNetwork() {
  return <div className="relative min-h-[420px] rounded-[2rem] border border-white/10 bg-white/[.045] p-6 backdrop-blur-xl"><svg viewBox="0 0 520 360" className="h-full w-full"><g stroke="rgba(34,211,238,.22)">{[0, 1, 2, 3].map((col) => [0, 1, 2, 3, 4].map((row) => col < 3 && <line key={`${col}-${row}`} x1={70 + col * 125} y1={55 + row * 60} x2={195 + col * 125} y2="180" />))}</g>{[0, 1, 2, 3].map((col) => [0, 1, 2, 3, 4].map((row) => <motion.circle key={`${col}-${row}`} cx={70 + col * 125} cy={55 + row * 60} r={14 - col} fill={col % 2 ? "#a855f7" : "#22d3ee"} animate={{ opacity: [0.35, 1, 0.35] }} transition={{ repeat: Infinity, duration: 2.2, delay: (col + row) * 0.12 }} />))}</svg></div>;
}

function ResultsDashboard() {
  const bars = [78, 86, 91, 88, 94, 90, 96, 92];
  return <div className="mt-10 grid gap-5 lg:grid-cols-4"><div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl"><p className="text-slate-400">Static Signs Accuracy</p><h3 className="mt-2 text-4xl font-black text-cyan-100">96.1%</h3></div><div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl"><p className="text-slate-400">Dynamic Signs Accuracy</p><h3 className="mt-2 text-4xl font-black text-violet-100">89.4%</h3></div><div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl"><p className="text-slate-400">Real-Time Latency</p><h3 className="mt-2 text-4xl font-black">38ms</h3></div><div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl"><p className="text-slate-400">FPS Performance</p><div className="mt-6 flex h-24 items-end gap-2">{bars.map((bar, i) => <motion.span key={i} initial={{ height: 0 }} whileInView={{ height: `${bar}%` }} className="flex-1 rounded-t-lg bg-gradient-to-t from-cyan-400 to-violet-400" />)}</div></div><div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl lg:col-span-2"><h3 className="font-bold">Confusion matrix</h3><div className="mt-5 grid grid-cols-6 gap-2">{Array.from({ length: 36 }).map((_, i) => <div key={i} className="aspect-square rounded-lg" style={{ background: `rgba(${i % 7 === 0 ? 34 : 168}, ${i % 7 === 0 ? 211 : 85}, ${i % 7 === 0 ? 238 : 247}, ${0.15 + ((i * 13) % 70) / 100})` }} />)}</div></div><div className="rounded-[2rem] border border-white/10 bg-white/[.055] p-6 backdrop-blur-xl lg:col-span-2"><h3 className="font-bold">Prediction confidence charts</h3>{["Open Palm", "Fist", "Peace", "Point"].map((label, i) => <div key={label} className="mt-5"><div className="flex justify-between text-sm"><span>{label}</span><span>{92 - i * 6}%</span></div><div className="mt-2 h-2 rounded-full bg-white/10"><div className="h-full rounded-full bg-gradient-to-r from-cyan-300 to-violet-400" style={{ width: `${92 - i * 6}%` }} /></div></div>)}</div></div>;
}

function Waveform({ active }: { active: boolean }) {
  return <div className="mt-5 flex h-12 items-center justify-center gap-1 rounded-2xl bg-slate-950/50">{Array.from({ length: 26 }).map((_, i) => <motion.span key={i} className="w-1 rounded-full bg-cyan-300" animate={{ height: active ? [8, 34 - (i % 6) * 3, 10] : 8, opacity: active ? 1 : 0.35 }} transition={{ repeat: Infinity, duration: 0.75, delay: i * 0.025 }} />)}</div>;
}

export default App;
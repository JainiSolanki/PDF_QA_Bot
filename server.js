const express = require("express");
const cors = require("cors");
const multer = require("multer");
const axios = require("axios");
const axiosRetry = require("axios-retry").default;
const path = require("path");
const crypto = require("crypto");
const rateLimit = require("express-rate-limit");
const crypto = require("crypto");
const { fileTypeFromFile } = require("file-type");
const fs = require("fs");

const app = express(); // Trust first proxy for rate limiting if behind a proxy
const session = require("express-session");
require("dotenv").config();

// ------------------------------------------------------------------
// CONFIGURATION
// ------------------------------------------------------------------
const API_REQUEST_TIMEOUT = parseInt(
  process.env.API_REQUEST_TIMEOUT || "45000",
  10
);

const MAX_RETRY_ATTEMPTS = parseInt(
  process.env.MAX_RETRY_ATTEMPTS || "3",
  10
);

// ------------------------------------------------------------------
// APP SETUP
// ------------------------------------------------------------------
app.set("trust proxy", 1);
app.use(cors());
app.use(express.json());

// ------------------------------------------------------------------
// SESSION (per-user chat history)
// ------------------------------------------------------------------
app.use(
  session({
    secret: "pdf-qa-bot-secret-key",
    resave: false,
    saveUninitialized: true,
    cookie: {
      secure: false,
      maxAge: 1000 * 60 * 60 * 24,
    },
  })
);

// ------------------------------------------------------------------
// AXIOS RETRY CONFIG
// ------------------------------------------------------------------
axiosRetry(axios, {
  retries: MAX_RETRY_ATTEMPTS,
  retryDelay: axiosRetry.exponentialDelay,
  retryCondition: (error) =>
    axiosRetry.isNetworkOrIdempotentRequestError(error) ||
    error.code === "ECONNABORTED" ||
    (error.response && error.response.status >= 500),
});

// ------------------------------------------------------------------
// RATE LIMITERS
// ------------------------------------------------------------------
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
});

const askLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 30,
});

const summarizeLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
});

const compareLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
});

// Storage for uploaded PDFs
const UPLOAD_DIR = path.resolve(__dirname, "uploads");

if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR);
}

// ------------------------------------------------------------------
// MULTER CONFIG
// ------------------------------------------------------------------
const SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];

const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const uniqueName = `${Date.now()}-${Math.round(
      Math.random() * 1e9
    )}${ext}`;
    cb(null, uniqueName);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (SUPPORTED_EXTENSIONS.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error("Unsupported file type"));
    }
  },
});

// ------------------------------------------------------------------
// ROUTE: UPLOAD
// ------------------------------------------------------------------
app.post("/upload", uploadLimiter, upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        error: "No file uploaded. Use form field name 'file'.",
      });
    }

    // Generate unique sessionId
    const sessionId = crypto.randomUUID();
    const filePath = path.join(__dirname, req.file.path);

    await axios.post(
      "http://localhost:5000/process-pdf",
      { filePath, session_id: sessionId },
      { timeout: API_REQUEST_TIMEOUT }
    );

    res.json({
      message: "File uploaded & processed successfully",
      sessionId,
    });
  } catch (err) {
    console.error("Upload failed:", err.message);
    res.status(500).json({ error: "Upload failed" });
  }
});

// ------------------------------------------------------------------
// ROUTE: ASK QUESTION
// ------------------------------------------------------------------
app.post("/ask", askLimiter, async (req, res) => {
  const { question, sessionId } = req.body;

  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId." });
  }

  if (!question || typeof question !== "string" || !question.trim()) {
    return res.status(400).json({ error: "Invalid question." });
  }

  try {
    if (!req.session.chatHistory) {
      req.session.chatHistory = [];
    }

    req.session.chatHistory.push({
      role: "user",
      content: question.trim(),
    });

    const response = await axios.post(
      "http://localhost:5000/ask",
      {
        question: question.trim(),
        session_id: sessionId,
        history: req.session.chatHistory,
      },
      { timeout: API_REQUEST_TIMEOUT }
    );

    req.session.chatHistory.push({
      role: "assistant",
      content: response.data.answer,
    });

    res.json(response.data);
  } catch (error) {
    console.error("Ask failed:", error.message);
    res.status(500).json({ error: "Error asking question" });
  }
});

// ------------------------------------------------------------------
// ROUTE: CLEAR HISTORY
// ------------------------------------------------------------------
app.post("/clear-history", (req, res) => {
  if (req.session) {
    req.session.chatHistory = [];
  }
  res.json({ message: "History cleared" });
});

// ------------------------------------------------------------------
// ROUTE: SUMMARIZE
// ------------------------------------------------------------------
app.post("/summarize", summarizeLimiter, async (req, res) => {
  const { sessionId } = req.body;

  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId." });
  }

  try {
    const response = await axios.post(
      "http://localhost:5000/summarize",
      { session_id: sessionId },
      { timeout: API_REQUEST_TIMEOUT }
    );

    res.json({ summary: response.data.summary });
  } catch (err) {
    console.error("Summarize failed:", err.response?.data || err.message);

    if (err.code === "ECONNABORTED") {
      return res.status(504).json({ error: "Summarization timed out" });
    }

    res.status(500).json({ error: "Error summarizing document" });
  }
});

// ------------------------------------------------------------------
// ROUTE: COMPARE
// ------------------------------------------------------------------
app.post("/compare", compareLimiter, async (req, res) => {
  const { sessionId } = req.body;
  if (!sessionId) {
    return res.status(400).json({ error: "Missing sessionId." });
  }

  try {
    const response = await axios.post(
      "http://localhost:5000/compare",
      req.body,
      { timeout: API_REQUEST_TIMEOUT }
    );

    res.json({ comparison: response.data.comparison });
  } catch (err) {
    console.error("Compare failed:", err.response?.data || err.message);
    res.status(500).json({ error: "Error comparing documents" });
  }
});

app.listen(4000, () => console.log("Backend running on http://localhost:4000"));

const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const cors = require('cors');
const fs = require('fs');
require('dotenv').config();
const path = require('path'); // Add path module for safer file paths

const app = express();

// CORS settings
app.use(cors({
  origin: [
    'http://localhost:3000', // Your local frontend
  ],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true
}));

app.use(express.json());

// Set up Multer for file upload
const upload = multer({ dest: 'uploads/' });

app.get('/', (req, res) => {
  res.send('Welcome to the Handwritten Answer Evaluation API!');
});

// Route to evaluate the image
app.post('/evaluate', upload.single('image'), (req, res) => {
  const imagePath = req.file.path;
  const predefinedAnswer = req.body.answer;
  const gcpCreds = process.env.GOOGLE_CREDENTIALS_PATH;

  // Ensure Python is accessible using the correct command
  const pythonCommand = process.platform === 'win32' ? 'python' : 'python3'; // Use 'python' for Windows

  // Spawn Python process to run the script
  const python = spawn(pythonCommand, [
    path.join(__dirname, 'evaluate.py'), // Ensure the full path to evaluate.py is used
    imagePath,
    gcpCreds.replace(/\\/g, '/'), // Cross-platform file path handling
    predefinedAnswer
  ]);

  let result = '';
  
  // Collect data from stdout
  python.stdout.on('data', (data) => {
    result += data.toString();
  });

  // Log stderr for debugging
  python.stderr.on('data', (data) => {
    console.error(`stderr: ${data}`);
  });

  // Handle process close (finish)
  python.on('close', (code) => {
    fs.unlinkSync(imagePath); // Clean up the uploaded image file
    if (code !== 0) {
      return res.status(500).json({ error: `Python script failed with exit code ${code}` });
    }

    try {
      const parsed = JSON.parse(result); // Parse the result from Python script
      res.json(parsed); // Send the parsed result back to the client
    } catch (err) {
      console.error('Error parsing Python output:', err);
      res.status(500).json({ error: 'Failed to parse Python output.' });
    }
  });
});

// Set the server to listen on a port
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));

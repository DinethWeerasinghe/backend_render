/**
 * HF Radio Blackout Forecasting API Server
 * ==========================================
 * 
 * Node.js backend that uses the Python pickle model to generate
 * HF radio blackout probability forecasts.
 * 
 * Endpoints:
 *   GET  /health                    - Health check
 *   POST /predict/single            - Single prediction
 *   POST /predict/batch             - Batch predictions from CSV
 *   POST /predict/upload            - Upload CSV and get predictions
 *   GET  /predict/csv/:filename     - Predict from CSV in forecasts folder
 *   GET  /model/info                - Get model information
 * 
 * Author: HF Blackout API
 * Date: January 2026
 */

const express = require('express');
const { spawn } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const multer = require('multer');
const cors = require('cors');

// Initialize Express
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Configure multer for file uploads
const upload = multer({
    dest: 'uploads/',
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Paths
const PYTHON_SCRIPT = path.join(__dirname, 'predict.py');
const MODEL_PATH = path.join(__dirname, 'hf_blackout_model.pkl');
const FORECASTS_DIR = path.join(__dirname, 'forecasts');

// Ensure directories exist
async function ensureDirectories() {
    try {
        await fs.mkdir(FORECASTS_DIR, { recursive: true });
        await fs.mkdir('uploads', { recursive: true });
        console.log('✓ Directories initialized');
    } catch (err) {
        console.error('Error creating directories:', err);
    }
}

/**
 * Execute Python script and return parsed JSON result
 */
async function executePython(args) {
    return new Promise((resolve, reject) => {
        const pythonPath = process.env.PYTHON_PATH || 'python';
        const python = spawn(pythonPath, [PYTHON_SCRIPT, ...args]);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        python.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        python.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python script failed: ${stderr}`));
                return;
            }

            try {
                const result = JSON.parse(stdout);
                resolve(result);
            } catch (err) {
                reject(new Error(`Failed to parse Python output: ${err.message}`));
            }
        });

        python.on('error', (err) => {
            reject(new Error(`Failed to start Python: ${err.message}`));
        });
    });
}

/**
 * Load model information from pickle file
 */
async function getModelInfo() {
    try {
        // Call Python to extract model info
        const result = await executePython(['info']);
        return result;
    } catch (err) {
        // Fallback: return basic info
        return {
            version: '1.0',
            model_type: 'LogisticRegression',
            features: 7,
            note: 'Model info extraction requires Python script update'
        };
    }
}

// ============================================================================
// API ENDPOINTS
// ============================================================================

/**
 * GET /health
 * Health check endpoint
 */
app.get('/health', async (req, res) => {
    try {
        // Check if model file exists
        await fs.access(MODEL_PATH);

        // Check if Python script exists
        await fs.access(PYTHON_SCRIPT);

        res.json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            model: 'available',
            python: 'available'
        });
    } catch (err) {
        res.status(500).json({
            status: 'unhealthy',
            error: err.message
        });
    }
});

/**
 * POST /predict/single
 * Make a single prediction
 * 
 * Body:
 *   {
 *     "datetime": "2026-02-01 12:00:00",
 *     "flare_probability": 0.75
 *   }
 */
app.post('/predict/single', async (req, res) => {
    try {
        const { datetime, flare_probability } = req.body;

        // Validate input
        if (!datetime || flare_probability === undefined) {
            return res.status(400).json({
                success: false,
                error: 'Missing required fields: datetime, flare_probability'
            });
        }

        if (flare_probability < 0 || flare_probability > 1) {
            return res.status(400).json({
                success: false,
                error: 'flare_probability must be between 0 and 1'
            });
        }

        // Execute Python prediction
        const result = await executePython([datetime, flare_probability.toString()]);

        if (!result.success) {
            return res.status(500).json(result);
        }

        res.json(result);

    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

/**
 * POST /predict/batch
 * Make batch predictions from JSON array
 * 
 * Body:
 *   {
 *     "forecasts": [
 *       {"datetime": "2026-02-01 12:00:00", "flare_probability": 0.75},
 *       {"datetime": "2026-02-01 13:00:00", "flare_probability": 0.65}
 *     ]
 *   }
 */
app.post('/predict/batch', async (req, res) => {
    try {
        const { forecasts } = req.body;

        if (!forecasts || !Array.isArray(forecasts) || forecasts.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'Missing or invalid forecasts array'
            });
        }

        // Create temporary CSV
        const tempCsvPath = path.join('uploads', `temp_${Date.now()}.csv`);

        // Write CSV
        let csvContent = 'DateTime,flare_probability\n';
        for (const forecast of forecasts) {
            if (!forecast.datetime || forecast.flare_probability === undefined) {
                return res.status(400).json({
                    success: false,
                    error: 'Each forecast must have datetime and flare_probability'
                });
            }
            csvContent += `${forecast.datetime},${forecast.flare_probability}\n`;
        }

        await fs.writeFile(tempCsvPath, csvContent);

        // Execute Python prediction
        const result = await executePython(['batch', tempCsvPath]);

        // Clean up temp file
        try {
            await fs.unlink(tempCsvPath);
        } catch (err) {
            console.error('Error deleting temp file:', err);
        }

        if (!result.success) {
            return res.status(500).json(result);
        }

        res.json(result);

    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

/**
 * POST /predict/upload
 * Upload CSV file and get predictions
 * 
 * Form data:
 *   file: CSV file with columns (DateTime or date+hour) and flare_probability
 */
app.post('/predict/upload', upload.single('file'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'No file uploaded'
            });
        }

        // Execute Python prediction
        const result = await executePython(['batch', req.file.path]);

        // Clean up uploaded file
        try {
            await fs.unlink(req.file.path);
        } catch (err) {
            console.error('Error deleting uploaded file:', err);
        }

        if (!result.success) {
            return res.status(500).json(result);
        }

        res.json(result);

    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

/**
 * GET /predict/csv/:filename
 * Make predictions from CSV file in forecasts folder
 * 
 * Example: GET /predict/csv/solar_flare_forecast.csv
 */
app.get('/predict/csv/:filename', async (req, res) => {
    try {
        const { filename } = req.params;

        // Security: prevent directory traversal
        if (filename.includes('..') || filename.includes('/')) {
            return res.status(400).json({
                success: false,
                error: 'Invalid filename'
            });
        }

        const csvPath = path.join(FORECASTS_DIR, filename);

        // Check if file exists
        try {
            await fs.access(csvPath);
        } catch (err) {
            return res.status(404).json({
                success: false,
                error: `File not found: ${filename}`
            });
        }

        // Execute Python prediction
        const result = await executePython(['batch', csvPath]);

        if (!result.success) {
            return res.status(500).json(result);
        }

        res.json(result);

    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

/**
 * GET /forecasts
 * List available CSV files in forecasts folder
 */
app.get('/forecasts', async (req, res) => {
    try {
        const files = await fs.readdir(FORECASTS_DIR);
        const csvFiles = files.filter(f => f.endsWith('.csv'));

        res.json({
            success: true,
            count: csvFiles.length,
            files: csvFiles
        });

    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

/**
 * GET /model/info
 * Get model information
 */
app.get('/model/info', async (req, res) => {
    try {
        // Load model package info from Python
        const pythonCode = `
import pickle
import json

with open('${MODEL_PATH}', 'rb') as f:
    pkg = pickle.load(f)

info = {
    'version': pkg['version'],
    'model_type': pkg['model_type'],
    'training_date': pkg['training_date'],
    'training_samples': pkg['training_samples'],
    'test_samples': pkg['test_samples'],
    'features': pkg['feature_names'],
    'performance': pkg['performance'],
    'solar_cycle_info': pkg['solar_cycle_info']
}

print(json.dumps(info))
`;

        // Write temp Python script
        const tempScript = path.join('uploads', `info_${Date.now()}.py`);
        await fs.writeFile(tempScript, pythonCode);

        // Execute
        const python = spawn('python3', [tempScript]);

        let stdout = '';
        let stderr = '';

        python.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        python.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        python.on('close', async (code) => {
            // Clean up
            try {
                await fs.unlink(tempScript);
            } catch (err) { }

            if (code !== 0) {
                return res.status(500).json({
                    success: false,
                    error: stderr
                });
            }

            try {
                const info = JSON.parse(stdout);
                res.json({
                    success: true,
                    model: info
                });
            } catch (err) {
                res.status(500).json({
                    success: false,
                    error: 'Failed to parse model info'
                });
            }
        });

    } catch (err) {
        res.status(500).json({
            success: false,
            error: err.message
        });
    }
});

/**
 * GET /
 * API documentation
 */
app.get('/', (req, res) => {
    res.json({
        name: 'HF Radio Blackout Forecasting API',
        version: '1.0.0',
        endpoints: {
            'GET /health': 'Health check',
            'POST /predict/single': 'Single prediction (body: {datetime, flare_probability})',
            'POST /predict/batch': 'Batch predictions (body: {forecasts: [{datetime, flare_probability}]})',
            'POST /predict/upload': 'Upload CSV and predict (multipart/form-data: file)',
            'GET /predict/csv/:filename': 'Predict from CSV in forecasts folder',
            'GET /forecasts': 'List available CSV files',
            'GET /model/info': 'Get model information'
        },
        examples: {
            single: {
                method: 'POST',
                url: '/predict/single',
                body: {
                    datetime: '2026-02-01 12:00:00',
                    flare_probability: 0.75
                }
            },
            batch: {
                method: 'POST',
                url: '/predict/batch',
                body: {
                    forecasts: [
                        { datetime: '2026-02-01 12:00:00', flare_probability: 0.75 },
                        { datetime: '2026-02-01 13:00:00', flare_probability: 0.65 }
                    ]
                }
            },
            csv: {
                method: 'GET',
                url: '/predict/csv/solar_flare_forecast.csv',
                description: 'Reads from forecasts/solar_flare_forecast.csv'
            }
        }
    });
});

// ============================================================================
// ERROR HANDLING
// ============================================================================

// 404 handler
app.use((req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        path: req.path
    });
});

// Global error handler
app.use((err, req, res, next) => {
    console.error('Error:', err);
    res.status(500).json({
        success: false,
        error: err.message || 'Internal server error'
    });
});

// ============================================================================
// START SERVER
// ============================================================================

async function startServer() {
    try {
        // Initialize directories
        await ensureDirectories();

        // Check if model exists
        try {
            await fs.access(MODEL_PATH);
            console.log('✓ Model file found:', MODEL_PATH);
        } catch (err) {
            console.warn('⚠ Model file not found:', MODEL_PATH);
            console.warn('  Please ensure hf_blackout_model.pkl is in the same directory');
        }

        // Check if Python script exists
        try {
            await fs.access(PYTHON_SCRIPT);
            console.log('✓ Python script found:', PYTHON_SCRIPT);
        } catch (err) {
            console.warn('⚠ Python script not found:', PYTHON_SCRIPT);
            console.warn('  Please ensure predict.py is in the same directory');
        }

        // Start server
        app.listen(PORT, () => {
            console.log('\n' + '='.repeat(70));
            console.log(' HF RADIO BLACKOUT FORECASTING API SERVER');
            console.log('='.repeat(70));
            console.log(`\n✓ Server running on port ${PORT}`);
            console.log(`\n📡 API Endpoints:`);
            console.log(`   http://localhost:${PORT}/`);
            console.log(`   http://localhost:${PORT}/health`);
            console.log(`   http://localhost:${PORT}/predict/single`);
            console.log(`   http://localhost:${PORT}/predict/batch`);
            console.log(`   http://localhost:${PORT}/predict/upload`);
            console.log(`   http://localhost:${PORT}/predict/csv/:filename`);
            console.log(`   http://localhost:${PORT}/forecasts`);
            console.log(`   http://localhost:${PORT}/model/info`);
            console.log(`\n📂 Directories:`);
            console.log(`   Forecasts: ${FORECASTS_DIR}`);
            console.log(`   Uploads:   ${path.join(__dirname, 'uploads')}`);
            console.log('\n' + '='.repeat(70));
            console.log('\nReady to accept requests!\n');
        });

    } catch (err) {
        console.error('Failed to start server:', err);
        process.exit(1);
    }
}

// Start the server
startServer();

// Handle graceful shutdown
process.on('SIGTERM', () => {
    console.log('\nReceived SIGTERM, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGINT', () => {
    console.log('\nReceived SIGINT, shutting down gracefully...');
    process.exit(0);
});

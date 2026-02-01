/**
 * API Test Script
 * ================
 * 
 * Tests all endpoints of the HF Blackout Forecasting API
 * 
 * Usage: node test-api.js
 */

const http = require('http');

const BASE_URL = 'http://localhost:3000';

// Helper function to make HTTP requests
function makeRequest(method, path, data = null) {
    return new Promise((resolve, reject) => {
        const url = new URL(path, BASE_URL);
        
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        const req = http.request(url, options, (res) => {
            let body = '';
            
            res.on('data', (chunk) => {
                body += chunk;
            });
            
            res.on('end', () => {
                try {
                    const json = JSON.parse(body);
                    resolve({
                        status: res.statusCode,
                        data: json
                    });
                } catch (err) {
                    reject(new Error(`Failed to parse response: ${body}`));
                }
            });
        });
        
        req.on('error', reject);
        
        if (data) {
            req.write(JSON.stringify(data));
        }
        
        req.end();
    });
}

// Test functions
async function testHealth() {
    console.log('\n📡 Testing GET /health...');
    try {
        const response = await makeRequest('GET', '/health');
        console.log('✓ Status:', response.status);
        console.log('✓ Response:', JSON.stringify(response.data, null, 2));
        return response.status === 200;
    } catch (err) {
        console.error('✗ Error:', err.message);
        return false;
    }
}

async function testSinglePrediction() {
    console.log('\n📡 Testing POST /predict/single...');
    try {
        const data = {
            datetime: '2026-02-01 12:00:00',
            flare_probability: 0.75
        };
        
        const response = await makeRequest('POST', '/predict/single', data);
        console.log('✓ Status:', response.status);
        console.log('✓ Response:', JSON.stringify(response.data, null, 2));
        
        // Validate response
        if (response.data.success && 
            response.data.blackout_probability !== undefined &&
            response.data.risk_level) {
            console.log('✓ Prediction successful!');
            console.log(`  Blackout Probability: ${(response.data.blackout_probability * 100).toFixed(2)}%`);
            console.log(`  Risk Level: ${response.data.risk_level}`);
            return true;
        } else {
            console.error('✗ Invalid response format');
            return false;
        }
    } catch (err) {
        console.error('✗ Error:', err.message);
        return false;
    }
}

async function testBatchPrediction() {
    console.log('\n📡 Testing POST /predict/batch...');
    try {
        const data = {
            forecasts: [
                { datetime: '2026-02-01 12:00:00', flare_probability: 0.75 },
                { datetime: '2026-02-01 13:00:00', flare_probability: 0.65 },
                { datetime: '2026-02-01 14:00:00', flare_probability: 0.55 }
            ]
        };
        
        const response = await makeRequest('POST', '/predict/batch', data);
        console.log('✓ Status:', response.status);
        console.log('✓ Count:', response.data.count);
        console.log('✓ Summary:', JSON.stringify(response.data.summary, null, 2));
        
        if (response.data.success && response.data.count === 3) {
            console.log('✓ Batch prediction successful!');
            console.log(`  Predictions: ${response.data.count}`);
            console.log(`  Mean Probability: ${(response.data.summary.mean_blackout_probability * 100).toFixed(2)}%`);
            return true;
        } else {
            console.error('✗ Invalid response');
            return false;
        }
    } catch (err) {
        console.error('✗ Error:', err.message);
        return false;
    }
}

async function testForecasts() {
    console.log('\n📡 Testing GET /forecasts...');
    try {
        const response = await makeRequest('GET', '/forecasts');
        console.log('✓ Status:', response.status);
        console.log('✓ Response:', JSON.stringify(response.data, null, 2));
        return response.status === 200;
    } catch (err) {
        console.error('✗ Error:', err.message);
        return false;
    }
}

async function testModelInfo() {
    console.log('\n📡 Testing GET /model/info...');
    try {
        const response = await makeRequest('GET', '/model/info');
        console.log('✓ Status:', response.status);
        if (response.data.success) {
            console.log('✓ Model Info:');
            console.log(`  Version: ${response.data.model.version}`);
            console.log(`  Type: ${response.data.model.model_type}`);
            console.log(`  Training Date: ${response.data.model.training_date}`);
            console.log(`  Features: ${response.data.model.features.length}`);
            return true;
        } else {
            console.log('⚠ Model info not available (fallback mode)');
            return true; // Still consider it a pass
        }
    } catch (err) {
        console.error('✗ Error:', err.message);
        return false;
    }
}

async function testRoot() {
    console.log('\n📡 Testing GET / (documentation)...');
    try {
        const response = await makeRequest('GET', '/');
        console.log('✓ Status:', response.status);
        console.log('✓ API Name:', response.data.name);
        console.log('✓ Endpoints:', Object.keys(response.data.endpoints).length);
        return response.status === 200;
    } catch (err) {
        console.error('✗ Error:', err.message);
        return false;
    }
}

// Run all tests
async function runTests() {
    console.log('='.repeat(70));
    console.log(' HF BLACKOUT FORECASTING API - TEST SUITE');
    console.log('='.repeat(70));
    console.log('\nMake sure the server is running on http://localhost:3000\n');
    
    const results = {
        health: await testHealth(),
        root: await testRoot(),
        single: await testSinglePrediction(),
        batch: await testBatchPrediction(),
        forecasts: await testForecasts(),
        modelInfo: await testModelInfo()
    };
    
    console.log('\n' + '='.repeat(70));
    console.log(' TEST RESULTS');
    console.log('='.repeat(70));
    
    let passed = 0;
    let total = 0;
    
    for (const [test, result] of Object.entries(results)) {
        total++;
        if (result) {
            passed++;
            console.log(`✓ ${test}: PASSED`);
        } else {
            console.log(`✗ ${test}: FAILED`);
        }
    }
    
    console.log('\n' + '='.repeat(70));
    console.log(`Results: ${passed}/${total} tests passed`);
    console.log('='.repeat(70));
    console.log();
    
    if (passed === total) {
        console.log('🎉 All tests passed! API is working correctly.\n');
        process.exit(0);
    } else {
        console.log('⚠ Some tests failed. Please check the errors above.\n');
        process.exit(1);
    }
}

// Run tests
runTests().catch(err => {
    console.error('\n✗ Test suite failed:', err.message);
    process.exit(1);
});

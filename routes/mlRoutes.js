const express = require('express');
const axios = require('axios');

const router = express.Router();
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://ml-engine:8000';

// @route   POST /api/forecast
// @desc    Proxy to Python ML Engine to get forecasting data
// @access  Public (as per user request during testing)
router.post('/forecast', async (req, res) => {
  try {
    // Expected incoming parameters based on user's schema
    const { 
      latitude, 
      longitude, 
      police_station, 
      junction_name, 
      timestamp, 
      vehicle_type, 
      violation_type 
    } = req.body;

    // Optional: Pre-process timestamp for temporal features if the ML engine expects it
    // Or just pass the unified ISO timestamp as requested
    const mlPayload = {
      geospatial: {
        latitude,
        longitude,
        police_station,
        junction_name
      },
      temporal: {
        timestamp: timestamp || new Date().toISOString()
      },
      contextual: {
        vehicle_type,
        violation_type
      }
    };

    // Forward the request to the Python FastAPI microservice
    const mlResponse = await axios.post(`${ML_SERVICE_URL}/predict`, mlPayload, {
      timeout: 5000 // 5 seconds timeout to ensure low-latency handling
    });

    res.json({
      success: true,
      data: mlResponse.data
    });
  } catch (error) {
    console.error('ML Service Error:', error.message);
    
    // Check if the error was a timeout or connection refused
    if (error.code === 'ECONNABORTED') {
      return res.status(504).json({ success: false, error: 'ML Engine request timed out' });
    }
    if (error.code === 'ECONNREFUSED') {
      return res.status(503).json({ success: false, error: 'ML Engine is currently unreachable' });
    }

    res.status(500).json({ success: false, error: 'Failed to process forecast request' });
  }
});

module.exports = router;

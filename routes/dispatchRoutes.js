const express = require('express');
const DispatchLog = require('../models/DispatchLog');

const router = express.Router();

// @route   POST /api/dispatch
// @desc    Log a new dispatch decision based on ML forecast
// @access  Public (for hackathon testing, usually would be Private)
router.post('/', async (req, res) => {
  try {
    const { officerId, location, context, predictionResult, actionTaken } = req.body;

    const newDispatchLog = new DispatchLog({
      officerId,
      location,
      context,
      predictionResult,
      actionTaken
    });

    const savedLog = await newDispatchLog.save();
    res.status(201).json({ success: true, data: savedLog });
  } catch (err) {
    console.error('Dispatch Log Error:', err.message);
    res.status(500).json({ success: false, error: 'Server error saving dispatch log' });
  }
});

// @route   GET /api/dispatch
// @desc    Get historical dispatch logs
// @access  Public
router.get('/', async (req, res) => {
  try {
    // Optionally populate the officer name using the referenced User model
    const logs = await DispatchLog.find()
      .populate('officerId', ['name', 'role'])
      .sort({ timestamp: -1 });

    res.json({ success: true, data: logs });
  } catch (err) {
    console.error('Fetch Dispatch Logs Error:', err.message);
    res.status(500).json({ success: false, error: 'Server error fetching logs' });
  }
});

module.exports = router;

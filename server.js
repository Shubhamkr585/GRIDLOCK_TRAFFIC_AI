require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

// Import routes
const authRoutes = require('./routes/authRoutes');
const mlRoutes = require('./routes/mlRoutes');
const dispatchRoutes = require('./routes/dispatchRoutes');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Environment variables
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI;

// MongoDB Connection
mongoose.connect(MONGO_URI)
  .then(() => console.log('✅ Connected to MongoDB Atlas successfully!'))
  .catch((err) => console.error('❌ Error connecting to MongoDB:', err));

// Register API Routes
app.use('/api/auth', authRoutes);
app.use('/api', mlRoutes); // registers /api/forecast
app.use('/api/dispatch', dispatchRoutes);

// Basic Health Check Route
app.get('/', (req, res) => {
  res.json({ message: 'Welcome to GridLock AI Backend API (Phase 2)' });
});

// Global Error Handler Middleware
app.use((err, req, res, next) => {
  console.error('Unhandled Error:', err.stack);
  res.status(500).json({ 
    success: false, 
    error: 'Internal Server Error',
    message: err.message 
  });
});

// Start Server
app.listen(PORT, () => {
  console.log(`🚀 Backend Server running on port ${PORT}`);
});

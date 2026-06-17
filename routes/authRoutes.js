const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const router = express.Router();

const JWT_SECRET = process.env.JWT_SECRET || 'gridlock-secret-key-1234';

// Register a new user
router.post('/register', async (req, res) => {
  try {
    const { name, email, password, role } = req.body;

    let user = await User.findOne({ email });
    if (user) {
      return res.status(400).json({ success: false, error: 'User already exists' });
    }

    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    user = new User({
      name,
      email,
      password: hashedPassword,
      role: role || 'Traffic Officer',
    });

    await user.save();

    const payload = { user: { id: user.id, role: user.role } };
    jwt.sign(payload, JWT_SECRET, { expiresIn: '1d' }, (err, token) => {
      if (err) throw err;
      res.status(201).json({ success: true, token, user: { id: user.id, name, email, role: user.role } });
    });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ success: false, error: 'Server error during registration' });
  }
});

// Login user
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ success: false, error: 'Invalid Credentials' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, error: 'Invalid Credentials' });
    }

    const payload = { user: { id: user.id, role: user.role } };
    jwt.sign(payload, JWT_SECRET, { expiresIn: '1d' }, (err, token) => {
      if (err) throw err;
      res.json({ success: true, token, user: { id: user.id, name: user.name, email, role: user.role } });
    });
  } catch (err) {
    console.error(err.message);
    res.status(500).json({ success: false, error: 'Server error during login' });
  }
});

module.exports = router;

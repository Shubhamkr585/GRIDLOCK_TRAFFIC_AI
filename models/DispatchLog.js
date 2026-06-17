const mongoose = require('mongoose');

const dispatchLogSchema = new mongoose.Schema({
  officerId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
  },
  location: {
    latitude: { type: Number, required: true },
    longitude: { type: Number, required: true },
    junction_name: { type: String },
    police_station: { type: String },
  },
  context: {
    vehicle_type: { type: String },
    violation_type: { type: String },
  },
  predictionResult: {
    predictedCongestion: { type: Number },
    severity: { type: String },
  },
  actionTaken: {
    type: String,
    default: 'Pending',
  },
  timestamp: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('DispatchLog', dispatchLogSchema);

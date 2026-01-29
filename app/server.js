const express = require("express");
const app = express();

app.use(express.json());

const PORT = process.env.PORT || 8080;

// In-memory store (MVP only)
const reservations = [];

/**
 * Health check
 * Required for Kubernetes readiness/liveness probes
 */
app.get("/health", (req, res) => {
  res.status(200).json({
    status: "ok",
    uptime: process.uptime(),
    timestamp: new Date().toISOString()
  });
});

/**
 * List reservations
 */
app.get("/reservations", (req, res) => {
  res.json({
    count: reservations.length,
    data: reservations
  });
});

/**
 * Create a reservation
 */
app.post("/reservations", (req, res) => {
  const { guestName, roomNumber, fromDate, toDate } = req.body;

  if (!guestName || !roomNumber || !fromDate || !toDate) {
    return res.status(400).json({
      error: "Missing required fields"
    });
  }

  const reservation = {
    id: reservations.length + 1,
    guestName,
    roomNumber,
    fromDate,
    toDate,
    createdAt: new Date().toISOString()
  };

  reservations.push(reservation);

  res.status(201).json(reservation);
});

app.listen(PORT, () => {
  console.log(`Hotel Reservation App running on port ${PORT}`);
});

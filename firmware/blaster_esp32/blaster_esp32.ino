// Blaster controller: 1 relay (replaces the fire microswitch) + SHOOT and DONE buttons.
// Wiring: docs/img/wiring.svg. Serial protocol (115200, newline-terminated): autobot/hardware.py.
//
//   SHOOT button → pulse relay + print "SHOOT"   (recording: labels + fires if the master switch is on)
//   DONE button  → print "TURN"
//   "H"          → heartbeat from the laptop
//   "F"          → pulse relay + print "FIRED", only if a heartbeat arrived in the last HEARTBEAT_MS
//                  (so a crashed laptop can't leave it firing)

const int RELAY_PIN = 26;
const int SHOOT_PIN = 25;   // button to GND
const int DONE_PIN = 33;    // button to GND
const bool RELAY_ACTIVE_LOW = true;   // most relay modules click when IN goes LOW; flip if yours doesn't
const unsigned long PULSE_MS = 150;   // tune until one pulse = one dart (50–300)
const unsigned long COOLDOWN_MS = 500;
const unsigned long HEARTBEAT_MS = 500;
const unsigned long DEBOUNCE_MS = 30;

unsigned long lastHeartbeat = 0, lastPulse = 0;

void relay(bool on) { digitalWrite(RELAY_PIN, (on != RELAY_ACTIVE_LOW) ? HIGH : LOW); }

bool pulse() {
  if (millis() - lastPulse < COOLDOWN_MS) return false;
  relay(true);
  delay(PULSE_MS);
  relay(false);
  lastPulse = millis();
  return true;
}

struct Button {
  int pin;
  bool last = HIGH;
  unsigned long changedAt = 0;
  // true once per press (HIGH → LOW, debounced)
  bool pressed() {
    bool now = digitalRead(pin);
    if (now != last && millis() - changedAt > DEBOUNCE_MS) {
      changedAt = millis();
      last = now;
      return now == LOW;
    }
    return false;
  }
};

Button shootBtn{SHOOT_PIN}, doneBtn{DONE_PIN};

void setup() {
  relay(false);  // set the level before enabling the output so it can't glitch on at boot
  pinMode(RELAY_PIN, OUTPUT);
  relay(false);
  pinMode(SHOOT_PIN, INPUT_PULLUP);
  pinMode(DONE_PIN, INPUT_PULLUP);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  if (shootBtn.pressed()) {
    Serial.println("SHOOT");  // the label is the press, whether or not the blaster has power
    pulse();
  }
  if (doneBtn.pressed()) Serial.println("TURN");

  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'H') lastHeartbeat = millis();
    else if (c == 'F') {
      if (millis() - lastHeartbeat > HEARTBEAT_MS) Serial.println("NO_HEARTBEAT");
      else if (pulse()) Serial.println("FIRED");
      else Serial.println("COOLDOWN");
    }
  }
}

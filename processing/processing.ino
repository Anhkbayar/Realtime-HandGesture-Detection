#include "pitches.h"

#define THUMB 40
#define INDEX 41
#define MIDDLE 42
#define RING 45
#define PINKY 47

#define BUZZER 37
#define RELAY 5

int fingerPins[5] = { THUMB, INDEX, MIDDLE, RING, PINKY };

int fingerNotes[5] = {
  NOTE_C4,
  NOTE_D4,
  NOTE_E4,
  NOTE_F4,
  NOTE_G4
};

int fingers[5] = { 0, 0, 0, 0, 0 };

String TARGET = "1,0,1,0,0";

bool relayTriggered = false;   // prevents repeated buzzing

void setup() {
  for (int i = 0; i < 5; i++) {
    pinMode(fingerPins[i], OUTPUT);
  }

  pinMode(BUZZER, OUTPUT);
  pinMode(RELAY, OUTPUT);

  Serial.begin(115200);
  Serial.println("---Initialized---");
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    data.trim();

    // Parse incoming "0,1,0,1,0"
    int index = 0;
    for (int i = 0; i < 5; i++) {
      int commaIndex = data.indexOf(',');

      if (commaIndex != -1) {
        fingers[i] = data.substring(0, commaIndex).toInt();
        data = data.substring(commaIndex + 1);
      } else {
        fingers[i] = data.toInt();
      }
    }

    // ----------------------------
    // 1. CONTROL LIGHTS ONLY
    // ----------------------------
    for (int i = 0; i < 5; i++) {
      digitalWrite(fingerPins[i], fingers[i]);
    }

    // ----------------------------
    // 2. CHECK TARGET GESTURE
    // ----------------------------
    String current = "";
    for (int i = 0; i < 5; i++) {
      current += String(fingers[i]);
      if (i < 4) current += ",";
    }

    if (current == TARGET && !relayTriggered) {
      Serial.println("TARGET DETECTED!");

      // Trigger relay
      digitalWrite(RELAY, HIGH);

      // Play buzzer melody ONCE
      for (int i = 0; i < 5; i++) {
        tone(BUZZER, fingerNotes[i], 150);
        delay(200);
      }
      noTone(BUZZER);

      relayTriggered = true; // prevent repeat
    }

    // Reset trigger when gesture changes
    if (current != TARGET) {
      relayTriggered = false;
      digitalWrite(RELAY, LOW);
    }
  }
}
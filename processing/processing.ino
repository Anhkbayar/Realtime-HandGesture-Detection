#include "pitches.h"

#define THUMB 40
#define INDEX 41
#define MIDDLE 42
#define RING 45
#define PINKY 47

#define BUZZER 37

int fingers[5] = { 0, 0, 0, 0, 0 };

int fingerNotes[5] = {
  NOTE_C4,  // Thumb
  NOTE_D4,  // Index
  NOTE_E4,  // Middle
  NOTE_F4,  // Ring
  NOTE_G4   // Pinky
};

void setup() {
  pinMode(THUMB, OUTPUT);
  pinMode(INDEX, OUTPUT);
  pinMode(MIDDLE, OUTPUT);
  pinMode(RING, OUTPUT);
  pinMode(PINKY, OUTPUT);

  pinMode(BUZZER, OUTPUT);

  Serial.begin(115200);
  Serial.println("---Initialized---");
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');

    int index = 0;
    int lastPos = 0;

    for (int i = 0; i < data.length(); i++) {
      if (data[i] == ',' || i == data.length() - 1) {
        String valueStr;

        if (i == data.length() - 1) {
          valueStr = data.substring(lastPos);
        } else {
          valueStr = data.substring(lastPos, i);
        }

        fingers[index] = valueStr.toInt();
        index++;
        lastPos = i + 1;
      }
    }

    // Apply to LEDs
    int fingerPins[5] = { THUMB, INDEX, MIDDLE, RING, PINKY };

    bool playing = false;

    for (int i = 0; i < 5; i++) {
      // LED control
      digitalWrite(fingerPins[i], fingers[i]);

      // Sound control
      if (fingers[i] == 1 && !playing) {
        tone(BUZZER, fingerNotes[i]);
        playing = true;
      }
    }

    if (!playing) {
      noTone(BUZZER);
    }

    // Debug
    Serial.print("Received: ");
    for (int i = 0; i < 5; i++) {
      Serial.print(fingers[i]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

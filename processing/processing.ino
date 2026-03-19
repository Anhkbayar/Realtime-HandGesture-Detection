#define THUMB 40
#define INDEX 41
#define MIDDLE 42
#define RING 45
#define PINKY 47

int fingers[5] = {0, 0, 0, 0, 0};

void setup() {
  pinMode(THUMB, OUTPUT);
  pinMode(INDEX, OUTPUT);
  pinMode(MIDDLE, OUTPUT);
  pinMode(RING, OUTPUT);
  pinMode(PINKY, OUTPUT);

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
    digitalWrite(THUMB, fingers[0]);
    digitalWrite(INDEX, fingers[1]);
    digitalWrite(MIDDLE, fingers[2]);
    digitalWrite(RING, fingers[3]);
    digitalWrite(PINKY, fingers[4]);

    // Debug
    Serial.print("Received: ");
    for (int i = 0; i < 5; i++) {
      Serial.print(fingers[i]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

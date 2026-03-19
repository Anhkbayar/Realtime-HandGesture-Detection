#define THUMB 40
#define INDEX 41
#define MIDDLE 42
#define RING 45
#define PINKY 47

int value = 0;

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
    value = Serial.parseInt();  // read integer

    Serial.print("Received: ");
    Serial.println(value);

    // Bit mapping
    digitalWrite(THUMB,  value & 1);        // bit 0
    digitalWrite(INDEX, (value >> 1) & 1);  // bit 1
    digitalWrite(MIDDLE,(value >> 2) & 1);  // bit 2
    digitalWrite(RING,  (value >> 3) & 1);  // bit 3
    digitalWrite(PINKY, (value >> 4) & 1);  // bit 4
  }
}
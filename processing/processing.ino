#define THUMB 40
#define INDEX 41
#define MIDDLE 42
#define RING 45
#define PINKY 47

char received;

void setup() {
  // put your setup code here, to run once:
  pinMode(THUMB, OUTPUT);
  pinMode(INDEX, OUTPUT);
  pinMode(MIDDLE, OUTPUT);
  pinMode(RING, OUTPUT);
  pinMode(PINKY, OUTPUT);

  digitalWrite(THUMB, LOW);
  digitalWrite(INDEX, LOW);
  digitalWrite(MIDDLE, HIGH);
  digitalWrite(RING, LOW);
  digitalWrite(PINKY, LOW);

  Serial.begin(115200);
  Serial.println("---Initialized---");
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available() > 0) {
    received = Serial.read();

    Serial.print("Received: ");
    Serial.println(received);

    if (received == '1') {
      openHand();
    } else if (received == '0') {
      closeHand();
    }
  }
}

void openHand() {
  digitalWrite(THUMB, HIGH);
  digitalWrite(INDEX, HIGH);
  digitalWrite(MIDDLE, HIGH);
  digitalWrite(RING, HIGH);
  digitalWrite(PINKY, HIGH);
}

void closeHand() {
  digitalWrite(THUMB, LOW);
  digitalWrite(INDEX, LOW);
  digitalWrite(MIDDLE, LOW);
  digitalWrite(RING, LOW);
  digitalWrite(PINKY, LOW);
}

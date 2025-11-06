// Define pins
const int trigPin = 9;
const int echoPin = 10;
const int buzzer = 7;

long duration;
int distance;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(buzzer, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  // Send pulse to trigger ultrasonic sensor
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // Measure echo time
  duration = pulseIn(echoPin, HIGH);

  // Calculate distance (in cm)
  distance = duration * 0.034 / 2;
  Serial.print("Distance: ");
  Serial.print(distance);
  Serial.println(" cm");

  // If object is near, turn on buzzer
  if (distance < 10) {
    digitalWrite(buzzer, HIGH);  // Buzzer ON
  } else {
    digitalWrite(buzzer, LOW);   // Buzzer OFF
  }

  delay(300);
}
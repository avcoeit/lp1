#include <Servo.h>  // Include Servo library to control servo motor

// ----------------------
// PIN CONNECTIONS
// ----------------------
// Servo Motor:
//   Signal -> Pin 6
//   VCC    -> 5V
//   GND    -> GND
//
// Ultrasonic Sensor (4-pin HC-SR04):
//   VCC    -> 5V
//   GND    -> GND
//   TRIG   -> Pin 7
//   ECHO   -> Pin 8
// ----------------------

Servo servo_6;  // Create a Servo object to control servo motor
int v_dist = 0; // Variable to store measured distance (in cm)

// Function to read distance from 4-pin ultrasonic sensor
long readUltrasonicDistance(int trigPin, int echoPin) {
  pinMode(trigPin, OUTPUT);     // Set trigger pin as output
  digitalWrite(trigPin, LOW);   // Ensure trigger pin is LOW
  delayMicroseconds(2);         // Short delay to stabilize sensor

  digitalWrite(trigPin, HIGH);  // Send a HIGH pulse for 10 microseconds
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);   // Stop the pulse

  pinMode(echoPin, INPUT);      // Set echo pin as input to read the return pulse
  long duration = pulseIn(echoPin, HIGH); // Measure the time of the returned pulse in microseconds

  // Convert duration to distance in cm
  long distance = duration * 0.0343 / 2; // Speed of sound formula: distance = (time * 343 m/s)/2
  return distance; // Return distance in cm
}

void setup() {
  servo_6.attach(6, 500, 2500); // Attach servo motor to Pin 6 with min/max pulse width
  Serial.begin(9600);            // Initialize serial monitor for debugging
}

void loop() {
  // Read distance from ultrasonic sensor (TRIG=7, ECHO=8)
  v_dist = readUltrasonicDistance(7, 8);

  // Print measured distance to Serial Monitor
  Serial.print("Distance (cm): ");
  Serial.println(v_dist);

  // Control servo based on distance
  if (v_dist <= 100) {           // If object is closer than 100 cm
    servo_6.write(180);          // Move servo to 180 degrees
  } else {                        // If object is farther than 100 cm
    servo_6.write(90);           // Keep servo at 90 degrees
  }

  delay(200);                     // Small delay for stability and to avoid rapid servo movement
}



//for 3 pin sensor 


#include <Servo.h>

// ----------------------
// PIN CONNECTIONS
// ----------------------
// Servo Motor:
//   Signal -> Pin 6
//   VCC    -> 5V
//   GND    -> GND
//
// Ultrasonic Sensor (3-pin):
//   SIG    -> Pin 7  (Trigger + Echo)
//   VCC    -> 5V
//   GND    -> GND
// ----------------------

Servo servo_6;  // Servo object
int v_dist = 0; // Variable to store distance

// Function to read distance from 3-pin ultrasonic sensor
long readUltrasonicDistance(int sigPin) {
  pinMode(sigPin, OUTPUT);    // Set SIG as output to send pulse
  digitalWrite(sigPin, LOW);  // Clear pulse
  delayMicroseconds(2);

  digitalWrite(sigPin, HIGH); // Send 10 us pulse
  delayMicroseconds(10);
  digitalWrite(sigPin, LOW);

  pinMode(sigPin, INPUT);     // Read echo on the same pin
  long duration = pulseIn(sigPin, HIGH); // Measure time for echo

  // Convert duration to distance in cm
  long distance = duration * 0.0343 / 2;
  return distance;
}

void setup() {
  servo_6.attach(6, 500, 2500); // Attach servo
  Serial.begin(9600);           // Initialize serial monitor
}

void loop() {
  v_dist = readUltrasonicDistance(7);  // Read distance from 3-pin sensor

  Serial.print("Distance (cm): ");
  Serial.println(v_dist);

  if (v_dist <= 100) {        // Move servo if object is close
    servo_6.write(180);
  } else {
    servo_6.write(90);
  }

  delay(200);                 // Small delay for stability
}

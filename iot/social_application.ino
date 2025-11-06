// ---------------------------
// PIN DIAGRAM & CONNECTIONS
// ---------------------------
//
// Ultrasonic Sensors (HC-SR04) for 4 lanes
//
//     [TRIG1] -> Pin 2      [ECHO1] -> Pin 3
//     [TRIG2] -> Pin 4      [ECHO2] -> Pin 5
//     [TRIG3] -> Pin 6      [ECHO3] -> Pin 7
//     [TRIG4] -> Pin 8      [ECHO4] -> Pin 9
//
// LEDs for lane signals
//
//     [LED1] -> Pin 10  (Lane 1)
//     [LED2] -> Pin 11  (Lane 2)
//     [LED3] -> Pin 12  (Lane 3)
//     [LED4] -> Pin 13  (Lane 4)

//  The long leg of LED is the anode (+) and connects to the Arduino pin.

//  The short leg of LED is the cathode (-) and connects to GND through the resistor.
//
// Note: Connect the GND of sensors and LEDs to Arduino GND, and VCC of sensors to 5V.
//
// ---------------------------
// Pin Definitions
// ---------------------------

// Ultrasonic Sensors
#define TRIG1 2
#define ECHO1 3
#define TRIG2 4
#define ECHO2 5
#define TRIG3 6
#define ECHO3 7
#define TRIG4 8
#define ECHO4 9

// LEDs for Lanes
#define LED1 10
#define LED2 11
#define LED3 12
#define LED4 13

// ---------------------------
// Function to get distance from ultrasonic sensor
// ---------------------------
long getDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);            // Clear trigger
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);           // Send pulse
  digitalWrite(trigPin, LOW);

  // Calculate distance in cm (speed of sound: 0.034 cm/us)
  long duration = pulseIn(echoPin, HIGH);
  long distance = duration * 0.034 / 2;
  return distance;
}

// ---------------------------
// Function to set traffic signal LEDs
// ---------------------------
void setSignal(int lane) {
  // Turn off all LEDs first
  digitalWrite(LED1, LOW);
  digitalWrite(LED2, LOW);
  digitalWrite(LED3, LOW);
  digitalWrite(LED4, LOW);

  // Turn ON only the selected lane's LED
  if (lane == 1) digitalWrite(LED1, HIGH);
  if (lane == 2) digitalWrite(LED2, HIGH);
  if (lane == 3) digitalWrite(LED3, HIGH);
  if (lane == 4) digitalWrite(LED4, HIGH);
}

// ---------------------------
// Arduino Setup
// ---------------------------
void setup() {
  Serial.begin(9600);

  // Set ultrasonic pins
  pinMode(TRIG1, OUTPUT); pinMode(ECHO1, INPUT);
  pinMode(TRIG2, OUTPUT); pinMode(ECHO2, INPUT);
  pinMode(TRIG3, OUTPUT); pinMode(ECHO3, INPUT);
  pinMode(TRIG4, OUTPUT); pinMode(ECHO4, INPUT);

  // Set LED pins
  pinMode(LED1, OUTPUT);
  pinMode(LED2, OUTPUT);
  pinMode(LED3, OUTPUT);
  pinMode(LED4, OUTPUT);
}

// ---------------------------
// Main Loop
// ---------------------------
void loop() {
  // Read distances from all sensors
  long d1 = getDistance(TRIG1, ECHO1);
  long d2 = getDistance(TRIG2, ECHO2);
  long d3 = getDistance(TRIG3, ECHO3);
  long d4 = getDistance(TRIG4, ECHO4);

  // Print distances for debugging
  Serial.print("Distances: ");
  Serial.print(d1); Serial.print(" | ");
  Serial.print(d2); Serial.print(" | ");
  Serial.print(d3); Serial.print(" | ");
  Serial.println(d4);

  // Determine which lane has the nearest vehicle
  int lane = 0;        // 0 means no car detected
  long minDist = 1000; // Arbitrary large value

  if(d1 < minDist && d1 < 30) { lane = 1; minDist = d1; }
  if(d2 < minDist && d2 < 30) { lane = 2; minDist = d2; }
  if(d3 < minDist && d3 < 30) { lane = 3; minDist = d3; }
  if(d4 < minDist && d4 < 30) { lane = 4; minDist = d4; }

  // Set the corresponding signal
  if(lane != 0) {
    setSignal(lane);
    Serial.print("Green for Lane: "); 
    Serial.println(lane);
  } else {
    setSignal(0); // No vehicles detected
    Serial.println("No vehicles detected.");
  }

  delay(5000); // Wait 5 seconds before next measurement
}

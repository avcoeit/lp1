// Pin connected to the red LED
int redLed = 9;

void setup() {
  // Set the LED pin as output
  pinMode(redLed, OUTPUT);
}

void loop() {
  // Turn the LED on
  digitalWrite(redLed, HIGH);
  delay(1000); // Wait for 1 second
  
  // Turn the LED off
  digitalWrite(redLed, LOW);
  delay(1000); // Wait for 1 second
}

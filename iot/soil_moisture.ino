#include <ESP8266WiFi.h>       // ESP8266 WiFi library
#include <ThingSpeak.h>        // ThingSpeak library for cloud communication

// ----------------------- WiFi Credentials -----------------------
const char* ssid = "Galaxy";       
const char* password = "omsairam";

// ----------------------- ThingSpeak Settings -----------------------
unsigned long channelID = 3088407;
const char* writeAPIKey = "XC5V1GOMDRFW4DB9";

// WiFi client for ThingSpeak
WiFiClient client;

// ----------------------- Sensor Pin -----------------------
const int soilMoisturePin = A0;   // Analog pin connected to soil moisture sensor

/*
    PIN DIAGRAM (ESP8266 with Soil Moisture Sensor)

           +-----------------+
           |     ESP8266     |
           |                 |
   3.3V ---| VIN         A0 |--- Analog Output (Sensor)
   GND ----| GND          D0|
   D1 ----|                |
           +----------------+

    Soil Moisture Sensor Pins:
    - VCC → 3.3V of ESP8266
    - GND → GND of ESP8266
    - AO  → A0 (Analog input) of ESP8266
*/

// ----------------------- Setup -----------------------
void setup() {
  Serial.begin(115200);         // Initialize serial monitor
  delay(10);

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");

  // Initialize ThingSpeak
  ThingSpeak.begin(client);
}

// ----------------------- Loop -----------------------
void loop() {
  // Read soil moisture sensor value
  int sensorValue = analogRead(soilMoisturePin);

  // Convert analog value to percentage
  int moisturePercent = map(sensorValue, 1024, 346, 0, 100);  // Adjust 346 based on your sensor calibration
  moisturePercent = constrain(moisturePercent, 0, 100);

  // Print values to Serial Monitor
  Serial.print("Raw Sensor Value: ");
  Serial.print(sensorValue);
  Serial.print(" | Soil Moisture: ");
  Serial.print(moisturePercent);
  Serial.println("%");

  // Send data to ThingSpeak
  ThingSpeak.setField(1, moisturePercent);
  int response = ThingSpeak.writeFields(channelID, writeAPIKey);
  if (response == 200) {
    Serial.println("Data sent to ThingSpeak successfully.");
  } else {
    Serial.print("Error sending data. HTTP response code: ");
    Serial.println(response);
  }

  // Wait 15 seconds before next reading
  delay(15000);
}

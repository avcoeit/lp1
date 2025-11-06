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

// ----------------------- Soil Moisture Sensor -----------------------
const int soilMoisturePin = A0;   // Analog output (AO) from LM393 module

/*
    PIN CONNECTION DIAGRAM

    ESP8266 NodeMCU        LM393 Soil Moisture Sensor Module
    ----------------       ----------------------
    3.3V    --------------> VCC       (Power supply)
    GND     --------------> GND       (Ground)
    A0      --------------> AO        (Analog output to measure moisture)
    D1      --------------> DO        (Optional digital output for wet/dry, if used)

    Soil Probe ----------> LM393 module probe pins (+ and -)
    - Insert probe into soil
    - AO reads voltage proportional to soil moisture
    - DO gives HIGH/LOW if threshold is reached (adjustable by potentiometer)
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
  // Adjust 1024 and 346 based on calibration for dry and wet soil
  int moisturePercent = map(sensorValue, 1024, 346, 0, 100);
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

  // Wait 15 seconds before next reading (ThingSpeak rate limit)
  delay(15000);
}

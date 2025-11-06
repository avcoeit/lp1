/*
  IoT Temperature & Humidity Monitoring with ESP8266 and ThingSpeak

  https://arduino.esp8266.com/stable/package_esp8266com_index.json
  http://arduino.esp8266.com/stable/package_esp8266com_index.json


  PIN DIAGRAM:

      DHT11 Sensor       ESP8266 (NodeMCU)
      -------------------------------
      VCC      --------> 3.3V
      GND      --------> GND
      DATA     --------> D5 (GPIO14)

  Notes:
  - Connect the DHT11 VCC to 3.3V (do not use 5V with NodeMCU)
  - DATA pin connects to D5 (defined in code as DHTPIN)
  - Use a 10kΩ pull-up resistor between DATA and VCC if needed for stability
  - ESP8266 will send data to ThingSpeak via WiFi
*/

#include <ESP8266WiFi.h>  // ESP8266 WiFi library
#include "DHT.h"          // DHT sensor library
#include "ThingSpeak.h"   // ThingSpeak library

// ThingSpeak channel settings
unsigned long myChannelNumber = 3081700;       
const char* myWriteAPIKey = "VWEAOGX46Y8V1Qpp";

// WiFi credentials
const char* ssid = "realme";       
const char* password = "shreyash";

// DHT sensor settings
#define DHTPIN D5           // Pin where the DHT sensor DATA is connected
#define DHTTYPE DHT11       // DHT 11 sensor

DHT dht(DHTPIN, DHTTYPE);    // Initialize DHT sensor
WiFiClient client;           // Initialize WiFi client

void setup() {
  Serial.begin(115200);          // Start serial communication
  delay(10);

  // Connect to WiFi
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");

  // Initialize ThingSpeak
  ThingSpeak.begin(client);

  // Initialize DHT sensor
  dht.begin();
}

void loop() {
  // Read humidity and temperature
  float h = dht.readHumidity();
  float t = dht.readTemperature();

  // Check if any reading failed
  if (isnan(h) || isnan(t)) {
    Serial.println("Failed to read from DHT sensor!");
    delay(2000);  // Wait before trying again
    return;
  }

  // Print values to Serial Monitor
  Serial.print("Temperature: ");
  Serial.print(t);
  Serial.print(" °C, Humidity: ");
  Serial.print(h);
  Serial.println(" %");

  // Set fields for ThingSpeak
  ThingSpeak.setField(1, t);  // Field 1 for Temperature
  ThingSpeak.setField(2, h);  // Field 2 for Humidity

  // Write to ThingSpeak
  int x = ThingSpeak.writeFields(myChannelNumber, myWriteAPIKey);

  if (x == 200) {
    Serial.println("Data sent to ThingSpeak successfully!");
  } else {
    Serial.println("Error sending data: " + String(x));
  }

  delay(20000); // Wait 20 seconds before next reading
}

#include "Arduino_RouterBridge.h"
#include <DHT.h>

#define DHTPIN  2
#define DHTTYPE DHT22

const int BTN_PIN = 3;
const int LED_R   = 9;    // red   = high risk
const int LED_Y   = 10;   // amber = medium risk
const int LED_G   = 11;   // green = low risk

DHT dht(DHTPIN, DHTTYPE);

unsigned long lastReading = 0;
const unsigned long READING_PERIOD_MS = 30000;

// ── Button state with debounce ──────────────────────────────
bool water_state = false;
int  last_btn_reading = HIGH;
int  btn_state        = HIGH;
unsigned long last_debounce_time = 0;
const unsigned long DEBOUNCE_MS = 50;

void updateButton() {
  int reading = digitalRead(BTN_PIN);
  if (reading != last_btn_reading) {
    last_debounce_time = millis();
  }
  if ((millis() - last_debounce_time) > DEBOUNCE_MS) {
    if (reading != btn_state) {
      btn_state = reading;
      if (btn_state == LOW) {            // falling edge = press
        water_state = !water_state;
        Serial.print("[btn] water_state -> ");
        Serial.println(water_state ? "YES" : "no");
      }
    }
  }
  last_btn_reading = reading;
}

void setLEDs(bool r, bool y, bool g) {
  digitalWrite(LED_R, r ? HIGH : LOW);
  digitalWrite(LED_Y, y ? HIGH : LOW);
  digitalWrite(LED_G, g ? HIGH : LOW);
}

// Print float as "X.YY" without depending on dtostrf or printf-float support.
void printFloat2(float v) {
  if (isnan(v)) { Serial.print("nan"); return; }
  if (v < 0)    { Serial.print("-"); v = -v; }
  int whole = (int)v;
  int frac  = (int)((v - whole) * 100.0f + 0.5f);
  if (frac >= 100) { whole++; frac -= 100; }
  Serial.print(whole);
  Serial.print(".");
  if (frac < 10) Serial.print("0");
  Serial.print(frac);
}

void setup() {
  pinMode(LED_R, OUTPUT);
  pinMode(LED_Y, OUTPUT);
  pinMode(LED_G, OUTPUT);
  pinMode(BTN_PIN, INPUT_PULLUP);

  Serial.begin(115200);
  dht.begin();
  Bridge.begin();

  // boot blink
  for (int i = 0; i < 3; i++) {
    setLEDs(true, true, true);  delay(150);
    setLEDs(false, false, false); delay(150);
  }
  setLEDs(false, false, true);  // green = ready
}

void loop() {
  updateButton();   // poll button every loop pass

  if (millis() - lastReading < READING_PERIOD_MS) return;
  lastReading = millis();

  float temp_c       = dht.readTemperature();
  float humidity_pct = dht.readHumidity();
  bool  standing_water = water_state;

  if (isnan(temp_c) || isnan(humidity_pct)) {
    Serial.println("DHT22 read failed");
    setLEDs(true, true, false);  // R+Y = sensor error
    return;
  }

Serial.print("Classifying: ");
printFloat2(temp_c);       Serial.print("C, ");
printFloat2(humidity_pct); Serial.print("%, water=");
Serial.println(standing_water ? "yes" : "no");

setLEDs(true, true, true);   // all three on = inference in progress

float risk_f = -1.0f;

RpcCall rpc = Bridge.call("classify", temp_c, humidity_pct, standing_water);

if (rpc.result(risk_f)) {
  int risk_code = (int)(risk_f + 0.5f);   // 0 = low, 1 = medium, 2 = high

  Serial.print("[result] risk_code=");
  Serial.println(risk_code);

  if      (risk_code == 0) setLEDs(false, false, true);   // green
  else if (risk_code == 1) setLEDs(false, true,  false);  // yellow
  else if (risk_code == 2) setLEDs(true,  false, false);  // red
  else                     setLEDs(true,  false, true);    // error
} else {
    Serial.print("[rpc error] code=");
    Serial.println(rpc.getErrorCode());
    Serial.print("[rpc error] msg=");
    Serial.println(rpc.getErrorMessage());
  
    setLEDs(true, false, true);  // error
}
}
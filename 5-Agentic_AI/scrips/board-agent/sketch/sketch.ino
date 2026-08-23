#include "Arduino_RouterBridge.h"
#include "Arduino_LED_Matrix.h"

ArduinoLEDMatrix matrix;

// 8 rows x 13 columns, 0 = off, 1 = on.
// Row 0 is the top row, column 0 is the left column.
uint8_t frame[8][13];

void clearFrame() {
  memset(frame, 0, sizeof(frame));
}

void showOff()   { clearFrame(); matrix.renderBitmap(frame, 8, 13); }

void showCheck() {
  clearFrame();
  int pts[][2] = { {5,2}, {6,3}, {7,4}, {6,5}, {5,6}, {4,7}, {3,8}, {2,9}, {1,10} };
  for (auto &p : pts) frame[p[0]][p[1]] = 1;
  matrix.renderBitmap(frame, 8, 13);
}

void showX() {
  clearFrame();
  // Both diagonals stay within columns 2..10 for rows 0..7, so no bounds
  // check is needed here. If you change the row count, re-check that.
  for (int i = 0; i < 8; i++) {
    frame[i][2 + i]  = 1;
    frame[i][10 - i] = 1;
  }
  matrix.renderBitmap(frame, 8, 13);
}

void showSmiley() {
  clearFrame();
  int pts[][2] = {
    {1,3},{1,4},{1,8},{1,9},                                 // eyes
    {5,2},{6,3},{6,4},{6,5},{6,6},{6,7},{6,8},{6,9},{5,10}   // smile
  };
  for (auto &p : pts) frame[p[0]][p[1]] = 1;
  matrix.renderBitmap(frame, 8, 13);
}

// A minimal 3x5 font for digits 0-9, drawn at rows 1-5, columns 5-7.
const uint8_t DIGIT_FONT[10][5] = {
  {0b111,0b101,0b101,0b101,0b111}, // 0
  {0b010,0b110,0b010,0b010,0b111}, // 1
  {0b111,0b001,0b111,0b100,0b111}, // 2
  {0b111,0b001,0b111,0b001,0b111}, // 3
  {0b101,0b101,0b111,0b001,0b001}, // 4
  {0b111,0b100,0b111,0b001,0b111}, // 5
  {0b111,0b100,0b111,0b101,0b111}, // 6
  {0b111,0b001,0b010,0b010,0b010}, // 7
  {0b111,0b101,0b111,0b101,0b111}, // 8
  {0b111,0b101,0b111,0b001,0b111}, // 9
};

const int DIGIT_ROW_OFFSET = 1;
const int DIGIT_COL_OFFSET = 5;

void showDigit(int d) {
  clearFrame();
  if (d < 0 || d > 9) return;
  for (int row = 0; row < 5; row++) {
    for (int col = 0; col < 3; col++) {
      if ((DIGIT_FONT[d][row] >> (2 - col)) & 1) {
        frame[row + DIGIT_ROW_OFFSET][col + DIGIT_COL_OFFSET] = 1;
      }
    }
  }
  matrix.renderBitmap(frame, 8, 13);
}

// ─── Bridge-exposed functions ───────────────────────────────────────

void set_builtin_led(bool state) {
  // The UNO Q's onboard LED is active-low: LOW lights it, HIGH turns it off.
  // Get this backwards and the LED is lit at boot and inverts every command.
  digitalWrite(LED_BUILTIN, state ? LOW : HIGH);
}

// pattern: "check" | "x" | "smiley" | "off" | "0".."9"
void set_led_matrix(String pattern) {
  if (pattern == "check")       showCheck();
  else if (pattern == "x")      showX();
  else if (pattern == "smiley") showSmiley();
  else if (pattern == "off")    showOff();
  else if (pattern.length() == 1 && isDigit(pattern[0])) showDigit(pattern.toInt());
  else                          showX();  // unrecognized pattern -> visible error state
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);   // active-low: HIGH means off
  matrix.begin();
  showOff();

  Bridge.begin();
  Bridge.provide_safe("set_builtin_led", set_builtin_led);
  Bridge.provide_safe("set_led_matrix", set_led_matrix);
}

void loop() {
  // All work is event-driven via Bridge calls from Python.
}

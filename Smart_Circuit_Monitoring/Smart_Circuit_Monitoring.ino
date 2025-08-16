#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <LiquidCrystal_I2C.h>
#include <Adafruit_BMP280.h>

LiquidCrystal_I2C lcd(0x27, 20, 4);
Adafruit_BMP280 bmp;

const int chipSelect = 10;
File dataFile;

void setup() {
  Serial.begin(9600);
  Wire.begin();
  lcd.init();
  lcd.backlight();

  if (!bmp.begin(0x76) && !bmp.begin(0x77)) {
    lcd.print("BMP280 error");
    while (1);
  }

  if (!SD.begin(chipSelect)) {
    lcd.print("SD fail");
    while (1);
  }

  if (!SD.exists("data.csv")) {
    dataFile = SD.open("data.csv", FILE_WRITE);
    if (dataFile) {
      dataFile.println("Pressure_bar,Temperature_C");
      dataFile.close();
    }
  }
  
  lcd.print("Logging Started");
}

void loop() {
  float pressureBar = bmp.readPressure() / 100000.0;
  float temperatureC = bmp.readTemperature();

  dataFile = SD.open("data.csv", FILE_WRITE);
  if (dataFile) {
    dataFile.print(pressureBar, 3);
    dataFile.print(",");
    dataFile.println(temperatureC, 1);
    dataFile.close();
  } else {
    lcd.clear();
    lcd.print("SD write error");
  }

  lcd.setCursor(0, 1);
  lcd.print("P:");
  lcd.print(pressureBar, 3);
  lcd.print(" bar   ");

  lcd.setCursor(0, 2);
  lcd.print("T:");
  lcd.print(temperatureC, 1);
  lcd.print(" C     ");

  delay(1000);
}

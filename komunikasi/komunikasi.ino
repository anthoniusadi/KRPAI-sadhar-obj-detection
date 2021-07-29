int incomingByte = 0; // for incoming serial data
long converted;
String result;
void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}


void loop() {

  while (Serial.available() > 0) {
    result = Serial.readStringUntil('\n');
    incomingByte = Serial.read();

    Serial.print("I received: ");
    Serial.println(result);
  }

}

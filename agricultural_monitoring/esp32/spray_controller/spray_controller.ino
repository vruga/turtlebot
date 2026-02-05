/**
 * ESP32 Sprinkler Controller for Agricultural Disease Detection
 *
 * Simple controller that receives commands from Raspberry Pi via serial
 * and activates the sprinkler via GPIO.
 *
 * Commands:
 *   PING          - Test connection (responds: PONG)
 *   SPRAY:<ms>    - Activate sprinkler for <ms> milliseconds
 *   STOP          - Emergency stop
 *   HEARTBEAT     - Keep-alive check
 *   STATUS        - Get current state
 *
 * Wiring:
 *   GPIO 23 -> Sprinkler control (pump/valve)
 *   GPIO 2  -> Built-in LED (status)
 *
 * Author: Agricultural Robotics Team
 */

#define SPRINKLER_PIN   23
#define LED_PIN         2
#define MAX_SPRAY_MS    10000
#define SERIAL_BAUD     115200

enum State { IDLE, SPRAYING, STOPPED };

State currentState = IDLE;
unsigned long sprayStart = 0;
unsigned long sprayDuration = 0;
String inputBuffer = "";

void setup() {
    Serial.begin(SERIAL_BAUD);
    pinMode(SPRINKLER_PIN, OUTPUT);
    pinMode(LED_PIN, OUTPUT);

    digitalWrite(SPRINKLER_PIN, LOW);
    digitalWrite(LED_PIN, HIGH);

    delay(1000);
    Serial.println("ESP32_SPRINKLER_READY");
}

void loop() {
    // Check spray completion
    if (currentState == SPRAYING) {
        if (millis() - sprayStart >= sprayDuration) {
            stopSpray();
            Serial.println("SPRAY_COMPLETE");
        }
    }

    // Read serial commands
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (inputBuffer.length() > 0) {
                processCommand(inputBuffer);
                inputBuffer = "";
            }
        } else {
            inputBuffer += c;
            if (inputBuffer.length() > 64) inputBuffer = "";
        }
    }

    // LED status
    digitalWrite(LED_PIN, currentState == SPRAYING ? HIGH : LOW);
    delay(1);
}

void processCommand(String cmd) {
    cmd.trim();
    cmd.toUpperCase();

    if (currentState == STOPPED && cmd != "RESUME") {
        Serial.println("ERROR:STOPPED");
        return;
    }

    if (cmd == "PING") {
        Serial.println("PONG");
    }
    else if (cmd == "HEARTBEAT") {
        Serial.println("HEARTBEAT_OK");
    }
    else if (cmd == "STATUS") {
        if (currentState == IDLE) Serial.println("STATUS:IDLE");
        else if (currentState == SPRAYING) Serial.println("STATUS:SPRAYING");
        else Serial.println("STATUS:STOPPED");
    }
    else if (cmd == "STOP") {
        stopSpray();
        currentState = STOPPED;
        Serial.println("STOPPED");
    }
    else if (cmd == "RESUME") {
        currentState = IDLE;
        Serial.println("RESUMED");
    }
    else if (cmd.startsWith("SPRAY:")) {
        int duration = cmd.substring(6).toInt();
        if (duration > 0 && duration <= MAX_SPRAY_MS) {
            startSpray(duration);
        } else {
            Serial.println("ERROR:INVALID_DURATION");
        }
    }
    else {
        Serial.println("ERROR:UNKNOWN");
    }
}

void startSpray(int duration) {
    if (currentState == SPRAYING) {
        Serial.println("ERROR:BUSY");
        return;
    }
    currentState = SPRAYING;
    sprayDuration = duration;
    sprayStart = millis();
    digitalWrite(SPRINKLER_PIN, HIGH);
    Serial.println("SPRAY_STARTED");
}

void stopSpray() {
    digitalWrite(SPRINKLER_PIN, LOW);
    if (currentState == SPRAYING) currentState = IDLE;
}

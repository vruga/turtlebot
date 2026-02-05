# ESP32 Sprinkler Controller

Simple ESP32 firmware for controlling the sprinkler via serial commands from the Raspberry Pi.

## Wiring

```
Raspberry Pi (USB) -----> ESP32 (USB)

ESP32 GPIO 23 -----> Sprinkler/Pump control
ESP32 GPIO 2  -----> Built-in LED (status)
```

## Serial Commands (115200 baud)

| Command | Response | Description |
|---------|----------|-------------|
| `PING` | `PONG` | Test connection |
| `SPRAY:<ms>` | `SPRAY_STARTED` then `SPRAY_COMPLETE` | Spray for N ms |
| `STOP` | `STOPPED` | Emergency stop |
| `RESUME` | `RESUMED` | Resume after stop |
| `STATUS` | `STATUS:IDLE/SPRAYING/STOPPED` | Current state |
| `HEARTBEAT` | `HEARTBEAT_OK` | Keep-alive |

## Upload

1. Install ESP32 board in Arduino IDE
2. Select "ESP32 Dev Module"
3. Upload `spray_controller.ino`

## Test

```bash
# On Mac/Linux
screen /dev/ttyUSB0 115200

# Type: PING
# Should see: PONG
```

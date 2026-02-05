# Agricultural Disease Detection System

Autonomous plant disease detection and treatment system for TurtleBot3 with Raspberry Pi 4B.

## Features

- **Real-time Disease Detection**: TensorFlow Lite CNN model optimized for Raspberry Pi
- **Automatic Spray Treatment**: ESP32-controlled spray nozzle system with safety interlocks
- **AI Recommendations**: Claude API integration for farmer-friendly advice
- **Web Dashboard**: Touch-friendly interface for field use
- **Non-blocking Architecture**: Teleop control remains responsive during inference

## System Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │───►│  Inference  │───►│   Spray     │
│  (ROS2)     │    │  (TFLite)   │    │  (ESP32)    │
└─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │
       │                  ▼                  │
       │          ┌─────────────┐           │
       └─────────►│  Dashboard  │◄──────────┘
                  │   (Flask)   │
                  └─────────────┘
                        │
                        ▼
                  ┌─────────────┐
                  │  Claude AI  │
                  │  (Advice)   │
                  └─────────────┘
```

## Quick Start

```bash
# 1. Setup
./setup.sh

# 2. Add your model
cp your_model.tflite models/plant_disease_model.tflite

# 3. Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 4. Test components
./scripts/test_components.sh

# 5. Start system
./scripts/start_system.sh
```

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Computer | Raspberry Pi 4B (4GB+ RAM) |
| Robot | TurtleBot3 Burger/Waffle |
| Camera | USB webcam (Lenovo or compatible) |
| Spray Controller | ESP32 development board |
| Relay | 5V relay module |
| Spray System | 12V solenoid valve + nozzle |

## Directory Structure

```
agricultural_monitoring/
├── config/              # YAML configuration files
├── models/              # TFLite model + conversion script
├── src/
│   ├── camera/          # Frame capture node
│   ├── inference/       # TFLite inference worker
│   ├── spray_control/   # ESP32 communication + safety
│   ├── llm/             # Claude API integration
│   └── dashboard/       # Flask web interface
├── launch/              # ROS2 launch files
├── scripts/             # Startup scripts
├── esp32/               # Arduino firmware
└── tests/               # Unit & integration tests
```

## Configuration

Edit files in `config/`:

- `model_config.yaml` - Model path, classes, thresholds
- `spray_config.yaml` - ESP32 port, spray durations, safety limits
- `llm_config.yaml` - API settings, prompt templates
- `system_config.yaml` - ROS2 topics, dashboard port

## Usage

### Keyboard Controls
- **SPACEBAR**: Capture frame for analysis
- **ESC**: Emergency stop

### Dashboard
Access at `http://<PI_IP>:8080`

- View live camera feed
- See detection history
- Read AI recommendations
- Manual spray control
- Emergency stop button

## Safety Features

1. **Max spray duration**: 10 seconds hard limit
2. **Cooldown period**: 2 seconds between sprays
3. **Hourly limit**: Maximum 20 sprays per hour
4. **Confidence threshold**: No spray below 80% confidence
5. **Heartbeat monitoring**: Emergency stop if ESP32 disconnects
6. **Manual confirmation**: Optional approval before spray

## Model Training

The system expects a TFLite model trained on plant disease images:

1. Train your Keras model on disease dataset
2. Convert to TFLite:
   ```bash
   python3 models/convert_to_tflite.py \
       --model your_model.h5 \
       --quantization int8 \
       --classes healthy early_blight late_blight ...
   ```

## ESP32 Setup

1. Open `esp32/spray_controller/spray_controller.ino` in Arduino IDE
2. Install ESP32 board support
3. Connect ESP32 via USB
4. Upload firmware
5. Wire relay to GPIO 23

See `esp32/README.md` for detailed wiring instructions.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not found | Check `/dev/video0` exists |
| Model error | Verify `.tflite` file path |
| ESP32 not responding | Check serial port, baud rate |
| Slow inference | Ensure TFLite runtime installed |
| Dashboard not loading | Check port 8080 not in use |

## Testing

```bash
# Run all tests
python3 -m pytest tests/ -v

# Test specific component
./scripts/test_components.sh camera
./scripts/test_components.sh model
./scripts/test_components.sh esp32
```

## License

MIT License - See LICENSE file

## Safety Warning

This system controls agricultural spray equipment. Always:
- Test with water before using chemicals
- Follow pesticide handling guidelines
- Never point spray at people or animals
- Ensure proper ventilation
- Keep emergency stop accessible

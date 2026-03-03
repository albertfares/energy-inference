#pragma once
// Linux-compatible INA3221 driver using /dev/i2c-* and ioctl
// Minimal API mirroring common parts of Rob Tillaart's Arduino library.
// Author: ChatGPT (ported for Linux)
// License: MIT

#include <cstdint>
#include <string>

// Forward-declare linux i2c structures without requiring callers to include linux headers.
struct i2c_smbus_ioctl_data;

class INA3221 {
public:
    // i2c_dev is typically "/dev/i2c-1"; address is usually 0x40..0x43 depending on A0/A1
    INA3221(const std::string& i2c_dev, uint8_t address = 0x40);
    ~INA3221();

    // Non-copyable
    INA3221(const INA3221&) = delete;
    INA3221& operator=(const INA3221&) = delete;

    // Open the device file and set the slave address.
    // Returns 0 on success, -errno on failure.
    int begin();

    // Close fd if open.
    void end();

    // Reset chip to default config (datasheet default mask 0xF127 used by many libs)
    int reset();

    // Configuration helpers
    int enableChannel(uint8_t ch);
    int disableChannel(uint8_t ch);
    int setAveraging(uint16_t avg_code);     // pass one of AVG_* below
    int setConversionTimeShunt(uint16_t code); // one of CT_* below
    int setConversionTimeBus(uint16_t code);   // one of CT_* below
    int setOperatingMode(uint16_t mode);     // one of MODE_* below

    // Measurements (millivolts, microvolts, milliamps)
    int readBusVoltageMV(uint8_t ch, int32_t& out_mv);
    int readShuntVoltageUV(uint8_t ch, int32_t& out_uv);

    // Set per-channel shunt resistor value in Ohms (default 0.1Ω).
    void setShuntResistor(uint8_t ch, double ohms);

    // Compute current in milliamps using stored shunt value.
    int readCurrentMA(uint8_t ch, double& out_ma);

    // Alert thresholds (microvolts). See datasheet: LSB=40uV left-justified by 3.
    int setWarningAlertUV(uint8_t ch, uint32_t uv);
    int setCriticalAlertUV(uint8_t ch, uint32_t uv);
    int getWarningAlertUV(uint8_t ch, uint32_t& uv);
    int getCriticalAlertUV(uint8_t ch, uint32_t& uv);

    // Die id/manufacturer registers
    int readManufacturerID(uint16_t& id);
    int readDieID(uint16_t& id);

    // Last error (negative errno-style)
    int last_error() const { return error_; }

    // Register field constants (public for convenience)
    enum : uint8_t {
        REG_CONFIGURATION   = 0x00,
        REG_SHUNT_VOLTAGE_1 = 0x01, // +2 per channel
        REG_BUS_VOLTAGE_1   = 0x02, // +2 per channel
        REG_MASK_ENABLE     = 0x0F,
        REG_POWER_VALID_HYS = 0x10,
        REG_POWER_VALID_UPP = 0x11,
        REG_WARN_LIMIT_1    = 0x12, // +2 per channel
        REG_CRIT_LIMIT_1    = 0x13, // +2 per channel
        REG_SHUNT_VOLT_SUM  = 0x0D,
        REG_SHUNT_VOLT_SUM_LIMIT = 0x0E,
        REG_MANUF_ID        = 0xFE,
        REG_DIE_ID          = 0xFF
    };

    // Bitfield helper values (match TI INA3221 datasheet)
    // Averaging (AVG bits 9..11)
    static constexpr uint16_t AVG_1    = 0 << 9;
    static constexpr uint16_t AVG_4    = 1 << 9;
    static constexpr uint16_t AVG_16   = 2 << 9;
    static constexpr uint16_t AVG_64   = 3 << 9;
    static constexpr uint16_t AVG_128  = 4 << 9;
    static constexpr uint16_t AVG_256  = 5 << 9;
    static constexpr uint16_t AVG_512  = 6 << 9;
    static constexpr uint16_t AVG_1024 = 7 << 9;

    // Conversion times (CT bits BUS:6..8, SHUNT:3..5)
    static constexpr uint16_t CT_140us  = 0;
    static constexpr uint16_t CT_204us  = 1;
    static constexpr uint16_t CT_332us  = 2;
    static constexpr uint16_t CT_588us  = 3;
    static constexpr uint16_t CT_1100us = 4;
    static constexpr uint16_t CT_2116us = 5;
    static constexpr uint16_t CT_4156us = 6;
    static constexpr uint16_t CT_8244us = 7;

    // Operating mode (MODE bits 0..2)
    static constexpr uint16_t MODE_POWER_DOWN      = 0;
    static constexpr uint16_t MODE_SHUNT_TRIGGERED = 1;
    static constexpr uint16_t MODE_BUS_TRIGGERED   = 2;
    static constexpr uint16_t MODE_SHUNT_BUS_TRIG  = 3;
    static constexpr uint16_t MODE_SHUNT_CONT      = 5;
    static constexpr uint16_t MODE_BUS_CONT        = 6;
    static constexpr uint16_t MODE_SHUNT_BUS_CONT  = 7;

private:
    int writeReg16(uint8_t reg, uint16_t value);
    int readReg16(uint8_t reg, uint16_t& value);
    static inline uint16_t swap16(uint16_t v) { return (uint16_t)((v << 8) | (v >> 8)); }

    int fd_ = -1;
    std::string dev_;
    uint8_t addr_;
    int error_ = 0;
    double shunt_ohm_[3] = {0.002f, 0.002f, 0.002f};
};

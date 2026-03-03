#include "ina3221_linux.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/i2c.h>
#include <sys/ioctl.h>
#include <unistd.h>

INA3221::INA3221(const std::string& i2c_dev, uint8_t address)
    : dev_(i2c_dev), addr_(address) {}

INA3221::~INA3221() { end(); }

int INA3221::begin() {
    fd_ = ::open(dev_.c_str(), O_RDWR);
    if (fd_ < 0) { error_ = -errno; return error_; }
    if (ioctl(fd_, I2C_SLAVE, addr_) < 0) { error_ = -errno; ::close(fd_); fd_ = -1; return error_; }
    return 0;
}

void INA3221::end() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

int INA3221::writeReg16(uint8_t reg, uint16_t value) {
    if (fd_ < 0) return error_ = -ENODEV;
    // INA3221 expects MSB then LSB after the register index.
    uint8_t buf[3];
    buf[0] = reg;
    buf[1] = static_cast<uint8_t>(value >> 8);     // MSB first
    buf[2] = static_cast<uint8_t>(value & 0xFF);   // then LSB
    ssize_t n = ::write(fd_, buf, 3);
    if (n != 3) { error_ = (n < 0) ? -errno : -EIO; return error_; }
    return 0;
}

int INA3221::readReg16(uint8_t reg, uint16_t& out) {
    if (fd_ < 0) return error_ = -ENODEV;
    // Write register pointer
    ssize_t nw = ::write(fd_, &reg, 1);
    if (nw != 1) { error_ = (nw < 0) ? -errno : -EIO; return error_; }
    uint8_t data[2];
    ssize_t nr = ::read(fd_, data, 2);
    if (nr != 2) { error_ = (nr < 0) ? -errno : -EIO; return error_; }
    out = static_cast<uint16_t>((data[0] << 8) | data[1]);
    return 0;
}

int INA3221::reset() {
    // Assert only the RESET bit (bit 15). Device reverts to default config.
    return writeReg16(REG_CONFIGURATION, 0x8000);
}

int INA3221::enableChannel(uint8_t ch) {
    if (ch > 2) return error_ = -EINVAL;
    uint16_t cfg;
    int rc = readReg16(REG_CONFIGURATION, cfg);
    if (rc) return rc;
    cfg |= static_cast<uint16_t>(1u << (14 - ch));  // CH_EN bits 14..12
    return writeReg16(REG_CONFIGURATION, cfg);
}

int INA3221::disableChannel(uint8_t ch) {
    if (ch > 2) return error_ = -EINVAL;
    uint16_t cfg;
    int rc = readReg16(REG_CONFIGURATION, cfg);
    if (rc) return rc;
    cfg &= static_cast<uint16_t>(~(1u << (14 - ch)));
    return writeReg16(REG_CONFIGURATION, cfg);
}

int INA3221::setAveraging(uint16_t code) {
    if (code > 7) return error_ = -EINVAL;        // accept 0..7 and shift internally
    uint16_t cfg;
    int rc = readReg16(REG_CONFIGURATION, cfg);
    if (rc) return rc;
    cfg &= static_cast<uint16_t>(~(7u << 9));     // AVG[11:9]
    cfg |= static_cast<uint16_t>((code & 7u) << 9);
    return writeReg16(REG_CONFIGURATION, cfg);
}

int INA3221::setConversionTimeShunt(uint16_t code) {
    if (code > 7) return error_ = -EINVAL;        // 0..7, VSHCT[5:3]
    uint16_t cfg;
    int rc = readReg16(REG_CONFIGURATION, cfg);
    if (rc) return rc;
    cfg &= static_cast<uint16_t>(~(7u << 3));
    cfg |= static_cast<uint16_t>((code & 7u) << 3);
    return writeReg16(REG_CONFIGURATION, cfg);
}

int INA3221::setConversionTimeBus(uint16_t code) {
    if (code > 7) return error_ = -EINVAL;        // 0..7, VBUSCT[8:6]
    uint16_t cfg;
    int rc = readReg16(REG_CONFIGURATION, cfg);
    if (rc) return rc;
    cfg &= static_cast<uint16_t>(~(7u << 6));
    cfg |= static_cast<uint16_t>((code & 7u) << 6);
    return writeReg16(REG_CONFIGURATION, cfg);
}

int INA3221::setOperatingMode(uint16_t mode) {
    if (mode > 7) return error_ = -EINVAL;        // MODE[2:0]
    uint16_t cfg;
    int rc = readReg16(REG_CONFIGURATION, cfg);
    if (rc) return rc;
    cfg &= static_cast<uint16_t>(~0x7u);
    cfg |= static_cast<uint16_t>(mode & 0x7u);
    return writeReg16(REG_CONFIGURATION, cfg);
}

int INA3221::readBusVoltageMV(uint8_t ch, int32_t& out_mv) {
    if (ch > 2) return error_ = -EINVAL;
    uint16_t raw;
    int rc = readReg16(static_cast<uint8_t>(REG_BUS_VOLTAGE_1 + ch * 2), raw);
    if (rc) return rc;

    // Bus-voltage register: SIGN in bit15, BD11..BD0 in bits [14:3], [2:0] reserved.
    // Arithmetic right shift to preserve sign, then scale: LSB = 8 mV.
    int16_t s = static_cast<int16_t>(raw);
    s >>= 3;                         // keep sign, drop [2:0]
    out_mv = static_cast<int32_t>(s) * 8;
    return 0;
}

int INA3221::readShuntVoltageUV(uint8_t ch, int32_t& out_uv) {
    if (ch > 2) return error_ = -EINVAL;
    uint16_t raw;
    int rc = readReg16(static_cast<uint8_t>(REG_SHUNT_VOLTAGE_1 + ch * 2), raw);
    if (rc) return rc;

    // Bits [14:3] contain signed data (SD11..SD0), [2:0] reserved.
    // Cast to int16_t first so the >> 3 is ARITHMETIC (sign-preserving).
    int16_t s = static_cast<int16_t>(raw);
    s >>= 3;                                  // arithmetic shift to drop [2:0]
    out_uv = static_cast<int32_t>(s) * 40;    // LSB = 40 µV
    return 0;
}

void INA3221::setShuntResistor(uint8_t ch, double ohms) {
    if (ch < 3) shunt_ohm_[ch] = ohms;
}

int INA3221::readCurrentMA(uint8_t ch, double& out_ma) {
    if (ch > 2) return error_ = -EINVAL;
    if (shunt_ohm_[ch] <= 0.0) return error_ = -EINVAL;  // avoid division by zero
    int32_t uv;
    int rc = readShuntVoltageUV(ch, uv);
    if (rc) return rc;
    double volts = static_cast<double>(uv) / 1e6f;       // µV -> V
    double current = volts / shunt_ohm_[ch];             // A
    out_ma = current * 1000.0;                           // mA
    return 0;
}

static inline uint16_t pack_limit_uv_rounded(uint32_t uv) {
    // Limit registers store value in bits [15:3] with LSB = 40 µV; [2:0] reserved.
    // Round to nearest LSB to avoid truncation bias.
    uint32_t ticks = (uv + 20u) / 40u;       // +20 for nearest rounding
    if (ticks > 0x1FFFu) ticks = 0x1FFFu;    // clamp to 13 bits
    return static_cast<uint16_t>((ticks & 0x1FFFu) << 3);
}

int INA3221::setWarningAlertUV(uint8_t ch, uint32_t uv) {
    if (ch > 2) return error_ = -EINVAL;
    // Datasheet full-scale ≈ 163.84 mV; clamp user input.
    if (uv > 163800u) uv = 163800u;
    uint16_t v = pack_limit_uv_rounded(uv);
    return writeReg16(static_cast<uint8_t>(REG_WARN_LIMIT_1 + ch * 2), v);
}

int INA3221::setCriticalAlertUV(uint8_t ch, uint32_t uv) {
    if (ch > 2) return error_ = -EINVAL;
    if (uv > 163800u) uv = 163800u;
    uint16_t v = pack_limit_uv_rounded(uv);
    return writeReg16(static_cast<uint8_t>(REG_CRIT_LIMIT_1 + ch * 2), v);
}

int INA3221::getWarningAlertUV(uint8_t ch, uint32_t& uv) {
    if (ch > 2) return error_ = -EINVAL;
    uint16_t raw;
    int rc = readReg16(static_cast<uint8_t>(REG_WARN_LIMIT_1 + ch * 2), raw);
    if (rc) return rc;
    uv = static_cast<uint32_t>(((raw >> 3) & 0x1FFFu) * 40u);
    return 0;
}

int INA3221::getCriticalAlertUV(uint8_t ch, uint32_t& uv) {
    if (ch > 2) return error_ = -EINVAL;
    uint16_t raw;
    int rc = readReg16(static_cast<uint8_t>(REG_CRIT_LIMIT_1 + ch * 2), raw);
    if (rc) return rc;
    uv = static_cast<uint32_t>(((raw >> 3) & 0x1FFFu) * 40u);
    return 0;
}

int INA3221::readManufacturerID(uint16_t& id) {
    return readReg16(REG_MANUF_ID, id);
}

int INA3221::readDieID(uint16_t& id) {
    return readReg16(REG_DIE_ID, id);
}

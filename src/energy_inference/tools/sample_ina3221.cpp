// sample_ina3221.cpp
// Compile: g++ -O3 -std=c++17 -Wall -Wextra -o sample_ina3221 sample_ina3221.cpp
// Run example:
//   ./sample_ina3221 --i2c /dev/i2c-1 --addr 0x40 --hw gpu --hz 500 --duration-ms 1000 --out power.csv
//   ./sample_ina3221 --i2c /dev/i2c-1 --addr 0x40 --hw cpu --hz 500 --duration-ms 1000 --out power.csv
//   ./sample_ina3221 --i2c /dev/i2c-1 --addr 0x40 --hw both --hz 500 --duration-ms 1000 --out power.csv
//   ./sample_ina3221 --i2c /dev/i2c-1 --addr 0x40 --hw gpu,io --hz 500 --duration-ms 1000 --out power.csv
//   ./sample_ina3221 --i2c /dev/i2c-1 --addr 0x40 --hw all --hz 500 --duration-ms 1000 --out power.csv
//
// (Note) This version uses the INA3221 I2C driver (ina3221_linux.h) instead of reading sysfs files.
//
// Minimal change goal (per your request):
//   - Remove iso_timestamp column and iso_now_ms() (you don’t need wall-clock time for alignment)
//   - Remove elapsed_ms column and elapsed computation
//   - Add mono_ns column (absolute CLOCK_MONOTONIC timestamp) for alignment with your Python t_start/t_end

#include <chrono>
#include <csignal>
#include <ctime>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/prctl.h>
#include <time.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cinttypes>  // PRIu64

#include "ina3221_linux.h"

static volatile sig_atomic_t g_stop = 0;
static void handle_sigint(int) { g_stop = 1; }

// Channel configuration
struct Channel {
    int id;
    int32_t bus_voltage_mv;
    double pwr;
};

typedef int (*csv_writer_t)(FILE*, int32_t, const Channel*, int);

void setConverstionTimeandAveraging(INA3221 &ina, double hz) {
    // Determine appropriate conversion time and averaging based on desired hz
    // Total conversion time per channel = (shunt CT + bus CT) * averaging
    // For simplicity, we use same CT for shunt and bus

    struct CtAvgOption {
        uint16_t ct_code;
        int ct_us;
        uint16_t avg_code;
        int avg_count;
    };

    const CtAvgOption options[] = {
        {INA3221::CT_140us, 140, INA3221::AVG_1, 1},
        {INA3221::CT_204us, 204, INA3221::AVG_4, 4},
        {INA3221::CT_332us, 332, INA3221::AVG_16, 16},
        {INA3221::CT_588us, 588, INA3221::AVG_64, 64},
        {INA3221::CT_1100us, 1100, INA3221::AVG_128, 128},
        {INA3221::CT_2116us, 2116, INA3221::AVG_256, 256},
        {INA3221::CT_4156us, 4156, INA3221::AVG_512, 512},
        {INA3221::CT_8244us, 8244, INA3221::AVG_1024, 1024},
    };

    // Default to the smallest (fastest) option.
    auto chosen = options[0];

    const double target_period_us = 1e6 / static_cast<double>(hz);

    for (const auto& opt : options) {
        const double total_time_us = (opt.ct_us * 2.0) * opt.avg_count * 3.0;
        if (total_time_us <= target_period_us) {
            chosen = opt;   // keep updating; last one that fits wins
        } else {
            break;          // assumes options are ordered from fast -> slow
        }
    }

    ina.setConversionTimeShunt(chosen.ct_code);
    ina.setConversionTimeBus(chosen.ct_code);
    ina.setAveraging(chosen.avg_code);

    std::cerr << "Set CT=" << chosen.ct_us << "us, AVG=" << chosen.avg_count
            << " for target hz=" << hz << "\n";
}

int main(int argc, char **argv) {
    // Automatically terminate when the parent process dies
    // Useful if the Python execution script crashes or is suspended
    prctl(PR_SET_PDEATHSIG, SIGINT);

    // Configuration for INA3221 device
    std::string i2c_dev = "/dev/i2c-1";
    int addr = 0x40;
    std::string hardware = "gpu";
    std::string out_csv = "power_timeseries.csv";
    double hz = 1000.0;
    long duration_ms = 10000000; // default 10000s

    // Optional: default shunt (Ohms). Jetson AGX Orin main INA3221 typically uses 2 mΩ.
    double shunt_ohm = 0.002;

    // simple CLI
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--i2c") { if (i+1 < argc) i2c_dev = argv[++i]; }
        else if (a == "--addr") { if (i+1 < argc) addr = std::strtol(argv[++i], nullptr, 0); }
        else if (a == "--hw" || a == "--hardware") { if (i+1 < argc) hardware = argv[++i]; }
        else if (a == "--out" || a == "--csv") { if (i+1 < argc) out_csv = argv[++i]; }
        else if (a == "--hz") { if (i+1 < argc) hz = std::stod(argv[++i]); }
        else if (a == "--interval-ms") { if (i+1 < argc) hz = 1000.0 / std::stod(argv[++i]); }
        else if (a == "--duration-ms") { if (i+1 < argc) duration_ms = std::stol(argv[++i]); }
        else if (a == "--shunt-ohm") { if (i+1 < argc) shunt_ohm = std::stod(argv[++i]); }
        else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: " << argv[0]
                << " [--i2c DEV] [--addr 0x40] [--hw gpu|cpu|io|both|all] [--out CSV] [--hz N]"
                << " [--duration-ms M] [--shunt-ohm R]\n";
            return 0;
        }
    }

    if (hz <= 0) { std::cerr << "hz must be > 0\n"; return 2; }

    // Determine which channels to read
    Channel channels[3];
    int num_channels = 0;
    bool has_io = false;
    // INA3221 has channels 0, 1, 2 (0-indexed)
    // Typically: channel 0 = GPU, channel 1 = CPU, channel 2 = 5V
    if (hardware == "gpu") {
        channels[num_channels++] = {0, 0, 0.0};
    } else if (hardware == "cpu") {
        channels[num_channels++] = {1, 0, 0.0};
    } else if (hardware == "io") {
        channels[num_channels++] = {2, 0, 0.0};
        has_io = true;
    } else if (hardware == "both") {
        channels[num_channels++] = {0, 0, 0.0};
        channels[num_channels++] = {1, 0, 0.0};
    } else if (hardware == "all") {
        channels[num_channels++] = {0, 0, 0.0};
        channels[num_channels++] = {1, 0, 0.0};
        channels[num_channels++] = {2, 0, 0.0};
        has_io = true;
    } else {
        std::cerr << "Unknown hardware type: " << hardware << " (use gpu, cpu, io, both, or all)\n";
        return 2;
    }

    const uint8_t addr_u8 = static_cast<uint8_t>(addr);

    // Open and configure INA3221
    INA3221 ina(i2c_dev, addr_u8);
    if (int rc = ina.begin()) {
        std::cerr << "Failed to open INA3221 on " << i2c_dev << " addr 0x"
                  << std::hex << int(addr_u8) << std::dec << " rc=" << rc << "\n";
        return 3;
    }

    // Basic configuration: reset, enable channels, set conversion/avg, continuous mode
    ina.reset(); // device defaults
    // originally for lower overhead, however, it increases the interval... weird
    // ina.disableChannel(2);
    ina.setConversionTimeShunt(INA3221::CT_1100us);
    ina.setConversionTimeBus(INA3221::CT_1100us);
    ina.setOperatingMode(INA3221::MODE_SHUNT_BUS_CONT);
    ina.setAveraging(INA3221::AVG_128);

    setConverstionTimeandAveraging(ina, hz);

    // Configure shunt resistors for channels we'll use
    for (int i = 0; i < num_channels; ++i) {
        ina.setShuntResistor(channels[i].id, shunt_ohm);
    }

    // Prepare output CSV with appropriate columns
    FILE *f = fopen(out_csv.c_str(), "w");
    if (!f) { std::cerr << "Failed to open output CSV: " << out_csv << "\n"; return 3; }

    // Write header based on what we're measuring
    // NOTE: iso_timestamp and elapsed_ms removed; mono_ns added for absolute monotonic alignment.
    fprintf(f, "mono_ns,bus_voltage_mV");
    if (has_io) {
        fprintf(f, ",io_voltage_mV");
    }
    const char *ch_names[] = {"gpu_power_mW", "cpu_power_mW", "io_power_mW"};
    for (int i = 0; i < num_channels; ++i) {
        fprintf(f, ",%s", ch_names[channels[i].id]);
    }
    fprintf(f, "\n");

    // IMPORTANT: allocate 8 MB buffer for output file to reduce overhead
    setvbuf(f, NULL, _IOFBF, 8 * 1024 * 1024);

    // signal handler
    struct sigaction sa{};
    sa.sa_handler = handle_sigint;
    sigaction(SIGINT, &sa, nullptr);

    // timing setup
    using namespace std::chrono;
    const double period_s = 1.0 / hz;
    struct timespec now_ts;
    clock_gettime(CLOCK_MONOTONIC, &now_ts);

    // align next wake to now + small delta to start immediately
    struct timespec next_ts = now_ts;
    // add 5ms offset to allow setup (optional)
    long add_ns = (long)(5e6);
    next_ts.tv_sec += (add_ns + next_ts.tv_nsec) / 1000000000;
    next_ts.tv_nsec = (next_ts.tv_nsec + add_ns) % 1000000000;

    // compute absolute end time if duration_ms > 0
    steady_clock::time_point end_time = steady_clock::time_point::max();
    if (duration_ms > 0) end_time = steady_clock::now() + milliseconds(duration_ms);

    // sampling loop
    uint64_t sample_idx = 0;
    while (!g_stop) {
        // absolute sleep until next_ts
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_ts, nullptr);

        // record absolute monotonic timestamp (nanoseconds since boot-ish)
        struct timespec ts_mono;
        clock_gettime(CLOCK_MONOTONIC, &ts_mono);
        uint64_t mono_ns = (uint64_t)ts_mono.tv_sec * 1000000000ULL + (uint64_t)ts_mono.tv_nsec;

        // Read bus voltage once (from first channel)
        int32_t bus_voltage_mv, io_voltage_mv;
        ina.readBusVoltageMV(channels[0].id, bus_voltage_mv);

        if (has_io) {
            ina.readBusVoltageMV(2, io_voltage_mv);
        }

        // Read all channels in tight loop (no branches)
        for (int i = 0; i < num_channels; ++i) {
            double ma = 0.0;
            if (ina.readCurrentMA(channels[i].id, ma) == 0) {
              if (channels[i].id == 2) {
                channels[i].pwr = io_voltage_mv * ma / 1000.0;
              } else {
                channels[i].pwr = bus_voltage_mv * ma / 1000.0;
              }
            } else {
                channels[i].pwr = 0.0;
            }
        }

        // write CSV row
        fprintf(f, "%" PRIu64 ",%d", mono_ns, bus_voltage_mv);
        if (has_io) {
            fprintf(f, ",%d", io_voltage_mv);
        }
        for (int i = 0; i < num_channels; ++i) {
            fprintf(f, ",%.6f", channels[i].pwr);
        }
        fprintf(f, "\n");

        sample_idx++;

        // break on duration
        if (steady_clock::now() >= end_time) break;

        long period_ns = static_cast<long>(period_s * 1e9);
        next_ts.tv_sec += (next_ts.tv_nsec + period_ns) / 1000000000;
        next_ts.tv_nsec = (next_ts.tv_nsec + period_ns) % 1000000000;
    }

    fflush(f);
    fclose(f);

    std::cerr << "Stopped. Wrote CSV: " << out_csv << " (" << sample_idx << " samples)\n";
    return 0;
}

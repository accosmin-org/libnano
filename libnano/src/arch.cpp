#include <libnan/arch.h>

#if defined(__APPLE__)
    #include <sys/sysctl.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <sys/sysinfo.h>
#else
    #error Unsupported platform
#endif

namespace nano
{
#if defined(__APPLE__)
    template <typename tinteger>
    tinteger sysctl_var(const char* name, const tinteger default_value)
    {
        tinteger value = 0;
        size_t size = sizeof(value);
        return sysctlbyname(name, &value, &size, nullptr, 0) ? default_value : value;
    }

    unsigned int logical_cpus()
    {
        return sysctl_var<unsigned int>("hw.logicalcpu", 0);
    }

    unsigned int physical_cpus()
    {
        return sysctl_var<unsigned int>("hw.physicalcpu", 0);
    }

    unsigned long long int memsize()
    {
        return sysctl_var<unsigned long long int>("hw.memsize", 0);
    }

#elif defined(__linux__)
    unsigned int logical_cpus()
    {
        return static_cast<unsigned int>(sysconf(_SC_NPROCESSORS_ONLN));
    }

    unsigned int physical_cpus()
    {
        unsigned int registers[4];
        __asm__ __volatile__ ("cpuid " :
            "=a" (registers[0]),
            "=b" (registers[1]),
            "=c" (registers[2]),
            "=d" (registers[3])
            : "a" (1), "c" (0));
        const unsigned CPUFeatureSet = registers[3];
        const bool hyperthreading = (CPUFeatureSet & (1u << 28)) != 0u;
        return hyperthreading ? (logical_cpus() / 2) : logical_cpus();
    }

    unsigned long long int memsize()
    {
        struct sysinfo info{};
        sysinfo(&info);
        return  static_cast<unsigned long long int>(info.totalram) *
                static_cast<unsigned long long int>(info.mem_unit);
    }
#endif
}

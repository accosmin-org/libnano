#include <nano/arch.h>

#if defined(__APPLE__)
    #include <sys/sysctl.h>
#elif defined(__linux__)
    #include <string>
    #include <cstdio>
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
        unsigned int threads_per_core = 1;

        FILE* fp = popen("lscpu | grep 'Thread' | cut -d ':' -f 2", "r");
        if (fp)
        {
            char text[16] = {'\0'};
            while (fgets(text, sizeof(text), fp))
            {
                try
                {
                    threads_per_core = static_cast<unsigned int>(std::stoi(text));
                }
                catch (std::exception&)
                {
                }
            }
        }

        const auto threads = logical_cpus();
        if (threads_per_core >= 1 && threads_per_core < threads)
        {
            return threads / threads_per_core;
        }
        else
        {
            return threads;
        }
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

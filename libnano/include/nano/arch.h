#pragma once

// export symbols in shared libraries
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #define NANO_PUBLIC __attribute__ ((dllexport))
    #else
        #define NANO_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #endif
    #define NANO_PRIVATE
#else
    #if __GNUC__ >= 4
        #define NANO_PUBLIC __attribute__ ((visibility ("default")))
        #define NANO_PRIVATE  __attribute__ ((visibility ("hidden")))
    #else
        #define NANO_PUBLIC
        #define NANO_PRIVATE
    #endif
#endif

// fix "unused variable" warnings
// todo: in C++17 these should be replaced with [[maybe_unused]] attribute
#define NANO_UNUSED1(x) (void)(x)
#define NANO_UNUSED2(x, y) NANO_UNUSED1(x); NANO_UNUSED1(y)
#define NANO_UNUSED3(x, y, z) NANO_UNUSED2(x, y); NANO_UNUSED1(z)
#define NANO_UNUSED4(x, y, z, u) NANO_UNUSED2(x, y); NANO_UNUSED2(z, u)

// fix "unused variable" warnings (only for release mode)
#ifdef NDEBUG
    #define NANO_UNUSED1_RELEASE(x) NANO_UNUSED1(x)
    #define NANO_UNUSED2_RELEASE(x, y) NANO_UNUSED2(x, y)
    #define NANO_UNUSED3_RELEASE(x, y, z) NANO_UNUSED3(x, y, z)
    #define NANO_UNUSED4_RELEASE(x, y, z, u) NANO_UNUSED4(x, y, z, u)
#else
    #define NANO_UNUSED1_RELEASE(x)
    #define NANO_UNUSED2_RELEASE(x, y)
    #define NANO_UNUSED3_RELEASE(x, y, z)
    #define NANO_UNUSED4_RELEASE(x, y, z, u)
#endif

namespace nano
{
    // system information
    NANO_PUBLIC unsigned int logical_cpus();
    NANO_PUBLIC unsigned int physical_cpus();
    NANO_PUBLIC unsigned long long int memsize();

    inline unsigned int memsize_gb()
    {
        const unsigned long long int giga = 1LL << 30;
        return static_cast<unsigned int>((memsize() + giga - 1) / giga);
    }
}

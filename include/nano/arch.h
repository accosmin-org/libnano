#pragma once

// export symbols in shared libraries
#if defined _WIN32 || defined __CYGWIN__
    #ifdef __GNUC__
        #ifndef NANO_PUBLIC
            #define NANO_PUBLIC __attribute__((dllexport))
        #else
            #define NANO_PUBLIC __attribute__((dllimport))
        #endif
    #else
        #ifndef NANO_PUBLIC
            #define NANO_PUBLIC __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
        #else
            #define NANO_PUBLIC __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
        #endif
    #endif
    #define NANO_PRIVATE
#else
    #if __GNUC__ >= 4
        #define NANO_PUBLIC  __attribute__((visibility("default")))
        #define NANO_PRIVATE __attribute__((visibility("hidden")))
    #else
        #define NANO_PUBLIC
        #define NANO_PRIVATE
    #endif
#endif

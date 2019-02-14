#include <iostream>
#include <nano/arch.h>
#include <nano/version.h>

int main(const int, const char* [])
{
    std::cout << "physical CPUs..." << nano::physical_cpus() << std::endl;
    std::cout << "logical CPUs...." << nano::logical_cpus() << std::endl;
    std::cout << "memsize........." << nano::memsize_gb() << "GB" << std::endl;
    std::cout << "version........." <<
        nano::major_version << "." << nano::minor_version << "." << nano::patch_version << std::endl;
    std::cout << "git hash........" << nano::git_commit_hash << std::endl;

    return EXIT_SUCCESS;
}

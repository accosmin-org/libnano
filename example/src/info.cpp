#include <iostream>
#include <nano/version.h>

int main(const int, const char* [])
{
    std::cout << "version........." <<
        nano::major_version << "." << nano::minor_version << "." << nano::patch_version << std::endl;
    std::cout << "git hash........" << nano::git_commit_hash << std::endl;

    return EXIT_SUCCESS;
}

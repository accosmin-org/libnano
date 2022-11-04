#include <fstream>
#include <nano/core/estimator.h>
#include <nano/core/stream.h>
#include <utest/utest.h>

using namespace nano;

template <typename testimator>
static auto check_stream(const testimator& estimator)
{
    // fails to serialize to closed files
    {
        std::ofstream stream;
        UTEST_CHECK_THROW(estimator.write(stream), std::runtime_error);
    }
    {
        std::ifstream stream;
        testimator    xestimator;
        UTEST_CHECK_THROW(xestimator.read(stream), std::runtime_error);
    }

    // serialization to and from in-memory blobs should work using class API
    string_t blob;
    {
        std::ostringstream stream;
        UTEST_CHECK_NOTHROW(estimator.write(stream));
        blob = stream.str();
    }
    {
        testimator         xestimator;
        std::istringstream stream(blob);
        UTEST_CHECK_NOTHROW(xestimator.read(stream));
        UTEST_CHECK_EQUAL(xestimator.parameters(), estimator.parameters());
    }

    // serialization to and from in-memory blobs should work using functional API
    {
        std::ostringstream ostream;
        UTEST_CHECK_NOTHROW(::nano::write(ostream, estimator));

        testimator         xestimator;
        std::istringstream istream(ostream.str());
        UTEST_CHECK_NOTHROW(::nano::read(istream, xestimator));
        UTEST_CHECK_EQUAL(xestimator.parameters(), estimator.parameters());
        return xestimator;
    }
}

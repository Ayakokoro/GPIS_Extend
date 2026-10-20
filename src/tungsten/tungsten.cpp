#include "Version.hpp"
#include "Shared.hpp"

#ifdef TUNGSTEN_ENABLE_CUDA
#include "PathTracerSeBatchBackend.hpp"
#include "integrators/path_tracer/PathTraceIntegrator.hpp"
#endif

using namespace Tungsten;

int main(int argc, const char *argv[])
{
#ifdef TUNGSTEN_ENABLE_CUDA
    setPathTracerGpBatchFactory(createPathTracerSeBatchBackend);
#endif
    CliParser parser("tungsten", "[options] scene1 [scene2 [scene3...]]");

    StandaloneRenderer renderer(parser, std::cout);

    parser.parse(argc, argv);

    if (parser.isPresent(OPT_VERSION)) {
        std::cout << "tungsten, version " << VERSION_STRING << std::endl;
        std::exit(0);
    }

    renderer.setup();

    while (renderer.renderScene());

    return 0;
}
